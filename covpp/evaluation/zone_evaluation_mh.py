import cv2 as cv
import numpy as np
from shapely.geometry import LineString
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig, PlanningConfig, RobotConfig
from covpp.base.geometry import MyPath
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.sequence_handler import SequencePaths
from covpp.evaluation.zone_evaluation_base import ZoneStatsBase, ZoneEvaluationBase


class ZoneEvaluation(ZoneEvaluationBase):
    """Concrete zone evaluation that computes timing, path, and area KPIs. """

    def __init__(self, zmd: ZoneMapData, paths: SequencePaths):
        super().__init__(zmd, paths)
        self.lg = MyLogger("ZoneEvaluation")

    def evaluate(self, times: dict = None):
        """Compute coverage statistics using map, generated paths, and timings.

        Args:
        - times: Dict with keys 'start', 'segmentation', 'coverage_planning', 'sequencing'.
        """
        self.stats = ZoneStatsBase()
        self.stats.map_name = self.zmd.orig_map_data.name
        self.stats.method = PipeLineConfig.method
        self.stats.ind = self.zmd.ind

        # validate data
        if not self.validate_data(times):
            self.save_result()
            raise ValueError("Data not valid for evaluation. Evaluation aborted.")

        self.stats.n_cells = self.paths.n_cells
        self.stats.n_cov_cells = self.paths.n_paths

        # calculate statistics
        self.calculate_time_statistics(times)
        self.calculate_path_statistics()
        self.calculate_travel_times()
        # calculate cleaned area statistics
        cleaned_acc = self.calculate_cleaned_accumulated_area()
        self.calculate_cleaning_area()
        self.calculate_cleaned_area_stats(cleaned_acc)
        self.update_map_planning(cleaned_acc)

        self.save_result()

    def calculate_time_statistics(self, times: dict) -> None:
        """Populate time-related statistics from timing checkpoints. """
        self.stats.ptime_segmentation = times['segmentation'] - times['start']
        self.stats.ptime_coverage_planning = times['coverage_planning'] - times['segmentation']
        self.stats.ptime_sequencing = times['sequencing'] - times['coverage_planning']
        self.stats.ptime_pipeline = times['sequencing'] - times['start']

    def calculate_path_statistics(self) -> None:
        """Compute path lengths and total yaw changes for full and coverage paths. """
        full_path: MyPath = self.paths.f_path_m

        # full path length
        dxy = np.diff(full_path.xy, axis=0)
        segment_lengths = np.sqrt((dxy ** 2).sum(axis=1))
        self.stats.full_length = float(np.sum(segment_lengths))

        # coverage path length
        coverage_length = 0.0
        for path in self.paths.paths_m:
            dxy = np.diff(path.xy, axis=0)
            segment_lengths = np.sqrt((dxy ** 2).sum(axis=1))
            coverage_length += np.sum(segment_lengths)
        self.stats.coverage_length = float(coverage_length)

        # full yaw changes
        dxy = np.diff(full_path.xy, axis=0)
        dxy = dxy[~np.all(dxy == 0, axis=1)]
        angles = np.arctan2(dxy[:, 1], dxy[:, 0])
        angle_diffs = np.diff(angles)
        angle_diffs = np.abs((angle_diffs + np.pi) % (2 * np.pi) - np.pi)
        yaw_changes = np.sum(np.abs(angle_diffs))
        self.stats.yaw_path_full = float(yaw_changes)

        # coverage yaw changes
        yaw_changes = 0.0
        for path in self.paths.paths_m:
            dxy = np.diff(path.xy, axis=0)
            dxy = dxy[~np.all(dxy == 0, axis=1)]
            if len(dxy) < 2:
                continue
            angles = np.arctan2(dxy[:, 1], dxy[:, 0])
            angle_diffs = np.diff(angles)
            angle_diffs = np.abs((angle_diffs + np.pi) % (2 * np.pi) - np.pi)
            yaw_changes += np.sum(np.abs(angle_diffs))
        yaw_changes = float(yaw_changes)
        self.stats.yaw_path_coverage = yaw_changes

    def calculate_travel_times(self):
        """Estimate travel times from path lengths and robot kinematic limits. """
        # full travel time
        t_lin = self.stats.full_length / RobotConfig.v_lin_avg
        t_ang = self.stats.yaw_path_full / RobotConfig.v_ang_avg
        self.stats.trav_time_full = t_lin + t_ang
        # coverage travel time
        t_lin = self.stats.coverage_length / RobotConfig.v_lin_avg
        t_ang = self.stats.yaw_path_coverage / RobotConfig.v_ang_avg
        self.stats.trav_time_coverage = t_lin + t_ang

    # ------------------------------------------------------------------------------------------------------------------
    # Cleaning area calculations
    # ------------------------------------------------------------------------------------------------------------------

    def calculate_cleaning_area(self):
        """Compute cleanable area in pixels and meters from ``map_cleaning``. """
        self.stats.cleaning_area_px = np.sum(self.zmd.map_cleaning == 255)
        self.stats.area_cleaning_m = self.stats.cleaning_area_px * (self.zmd.resolution**2)

    def calculate_cleaned_area_stats(self, cleaned_acc: np.ndarray) -> None:
        """Compute cleaned/overlap/uncleaned areas and coverage percentage. """

        cleaned_area_px = np.sum(cleaned_acc > 0)
        area_cleaned_m = cleaned_area_px * (self.zmd.resolution ** 2)
        self.stats.cleaned_area_px = cleaned_area_px
        self.stats.area_cleaned_m = area_cleaned_m

        # overlaped_area
        # calculate cleaned area based on the length of the path and cleaning width
        cleaning_width_m = 2 * PlanningConfig.cleaning_r
        total_cleaned_area_m = self.stats.coverage_length * cleaning_width_m  # full_length or coverage_length # imp
        self.stats.area_overlaped_m = total_cleaned_area_m - self.stats.area_cleaned_m
        area_overlap_pc = self.stats.area_overlaped_m / self.stats.area_cleaning_m * 100
        self.stats.area_overlap_pc = max(0.0, area_overlap_pc)  # avoid negative overlap due to rounding errors

        # area_uncleaned_m
        self.stats.area_uncleaned_m = self.stats.area_cleaning_m - self.stats.area_cleaned_m

        # coverage percentage
        self.stats.area_coverage_pc = self.stats.area_cleaned_m / self.stats.area_cleaning_m * 100

    def calculate_cleaned_accumulated_area(self) -> np.ndarray:
        """Accumulate cleaned passes per pixel by buffering coverage paths. """

        # [cleaned_acc]: accumulate number of passes for each pixel from paths
        # for covered area calculation

        # data
        cleaning_r_px = int(PlanningConfig.cleaning_r / self.zmd.resolution)  # cleaning radius in pixels
        g_xy = self.zmd.global_xy  # global offset

        #
        map_cleaning = self.zmd.map_cleaning.copy()     # map of cleanable area, from zone map data (without any paths)
        mask_free = map_cleaning == 255                 # free space mask in map_cleaning
        cleaned_acc = np.zeros_like(map_cleaning, dtype=np.uint8)

        # ----------------------------------------------------------------------
        # version 1: only coverage paths
        paths = self.paths.paths_px
        for path in paths:
            # # create buffered area around path
            # path_points = path.xy - g_xy
            # line = LineString(path_points)
            # buffered_poly = line.buffer(cleaning_r_px, cap_style=2)
            # buffered_poly = np.array(buffered_poly.exterior.coords).astype(int)

            # # Draw filled polygon onto image
            # cleaned_acc = cv.fillPoly(cleaned_acc, [buffered_poly], color=1)

            for i in range(path.n - 1):
                pt1 = tuple(map(int, path.point(i).xy - g_xy))
                pt2 = tuple(map(int, path.point(i + 1).xy - g_xy))
                cv.line(cleaned_acc, pt1, pt2, color=1, thickness=2 * cleaning_r_px)
                # add circle at the end of the line segment to ensure coverage
                cv.circle(cleaned_acc, pt1, cleaning_r_px, color=1, thickness=-1)
                cv.circle(cleaned_acc, pt2, cleaning_r_px, color=1, thickness=-1)

        # consider only free space pixels
        cleaned_acc = cleaned_acc * mask_free
        return cleaned_acc

    def update_map_planning(self, cleaned_acc: np.ndarray) -> None:
        """Update planning visualization with cleaning coverage overlays. """

        # [map_cleaned]: show uncleaned area and overlaps, only for visualization
        # uncleaned free pixels -> red, cleaned -> scaled green

        mask_free = self.zmd.map_cleaning == 255
        max_overlap = np.max(cleaned_acc)  # maximum overlap count among pixels

        d_c = int(np.floor(250 / max_overlap))
        map_cleaned = np.zeros((*self.zmd.map_cleaning.shape, 3), dtype=np.uint8)
        mask_not_cleaned = mask_free & (cleaned_acc == 0)
        mask_cleaned = mask_free & (cleaned_acc > 0)
        map_cleaned[mask_not_cleaned] = (0, 0, 200)

        if np.any(mask_cleaned):
            green = np.clip(d_c * cleaned_acc, 0, 255).astype(np.uint8)
            map_cleaned[..., 1][mask_cleaned] = green[mask_cleaned]

        self.zmd.map_cleaned = map_cleaned

        # --------------------------------------------------------------------------------------------------------------
        # # update map planning with cleaning paths
        # c = 40
        # color = (c, c, 0)
        # map_p_c = self.zmd.map_planning.copy()
        # # for path in paths:
        # for i in range(f_path.n - 1):
        #     pt1 = tuple(map(int, f_path.point(i).xy))
        #     pt2 = tuple(map(int, f_path.point(i+1).xy))
        #     cv.line(map_p_c, pt1, pt2, color=color, thickness=2 * cleaning_r_px)
        # alpha = 0.6
        # self.zmd.map_planning = cv.addWeighted(self.zmd.map_planning, alpha, map_p_c, 1 - alpha, 0)
