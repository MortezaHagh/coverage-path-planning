import cv2 as cv
import numpy as np
from shapely.geometry import LineString
from covpp.base.logger import MyLogger
from covpp.base.geometry import MyPath
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.sequence_handler import SequencePaths
from covpp.evaluation.zone_evaluation_base import ZoneStatsBase, ZoneEvaluationBase


class ZoneEvaluation(ZoneEvaluationBase):
    """Zone evaluation class to evaluate zone statistics. """

    def __init__(self, zmd: ZoneMapData, paths: SequencePaths):
        super().__init__(zmd, paths)
        self.lg = MyLogger("ZoneEvaluation")

    def evaluate(self, times: dict = None):
        """ Compute zone coverage statistics based on map, generated paths, and timing information. """
        self.stats = ZoneStatsBase()
        self.stats.map_name = self.zmd.orig_map_data.name
        self.stats.method = self.dcfg.method
        self.stats.ind = self.zmd.ind

        # validate data
        if not self.validate_data(times):
            self.save_result()
            raise ValueError("Data not valid for evaluation. Evaluation aborted.")

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
        """ calculate time statistics from timing information. """
        self.stats.time_segmentation = times['segmentation'] - times['start']
        self.stats.time_coverage_planning = times['coverage_planning'] - times['segmentation']
        self.stats.time_sequencing = times['sequencing'] - times['coverage_planning']
        self.stats.time_total = times['sequencing'] - times['start']

    def calculate_path_statistics(self) -> None:
        """ calculate path statistics: lengths and yaw changes. """
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
        self.stats.full_yaw_rad = float(yaw_changes)

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
        self.stats.coverage_yaw_rad = yaw_changes

    def calculate_travel_times(self):
        """ calculate estimated travel times based on path lengths and robot configurations. """
        # full travel time
        t_lin = self.stats.full_length / self.robot_cfgs.v_lin_avg
        t_ang = self.stats.full_yaw_rad / self.robot_cfgs.v_ang_avg
        self.stats.full_trav_time = t_lin + t_ang
        # coverage travel time
        t_lin = self.stats.coverage_length / self.robot_cfgs.v_lin_avg
        t_ang = self.stats.coverage_yaw_rad / self.robot_cfgs.v_ang_avg
        self.stats.coverage_trav_time = t_lin + t_ang

    # ------------------------------------------------------------------------------------------------------------------
    # Cleaning area calculations
    # ------------------------------------------------------------------------------------------------------------------

    def calculate_cleaning_area(self):
        """ calculate area to be cleaned in pixels and meters. """
        self.stats.cleaning_area_px = np.sum(self.zmd.map_cleaning == 255)
        self.stats.cleaning_area_m = self.stats.cleaning_area_px * (self.zmd.resolution**2)

    def calculate_cleaned_area_stats(self, cleaned_acc: np.ndarray) -> None:
        """ calculate cleaned area statistics from cleaned accumulated area. """

        cleaned_area_px = np.sum(cleaned_acc > 0)
        cleaned_area_m = cleaned_area_px * (self.zmd.resolution ** 2)
        self.stats.cleaned_area_px = cleaned_area_px
        self.stats.cleaned_area_m = cleaned_area_m

        # overlaped_area
        # calculate cleaned area based on the length of the path and cleaning width
        cleaning_width_m = 2 * self.pcfg.cleaning_r
        total_cleaned_area_m = self.stats.coverage_length * cleaning_width_m  # full_length or coverage_length # imp
        self.stats.overlaped_area_m = total_cleaned_area_m - self.stats.cleaned_area_m
        overlap_p = self.stats.overlaped_area_m / self.stats.cleaning_area_m * 100
        self.stats.overlap_p = max(0.0, overlap_p)  # avoid negative overlap due to rounding errors

        # uncleaned_area_m
        self.stats.uncleaned_area_m = self.stats.cleaning_area_m - self.stats.cleaned_area_m

        # coverage percentage
        self.stats.coverage_p = self.stats.cleaned_area_m / self.stats.cleaning_area_m * 100

    def calculate_cleaned_accumulated_area(self) -> np.ndarray:
        """ calculate cleaned accumulated area from paths. """

        # [cleaned_acc]: accumulate number of passes for each pixel from paths
        # for covered area calculation

        # data
        cleaning_r_px = int(self.pcfg.cleaning_r / self.zmd.resolution)  # cleaning radius in pixels
        g_xy = self.zmd.global_xy  # global offset

        #
        map_cleaning = self.zmd.map_cleaning.copy()     # map of cleanable area, from zone map data (without any paths)
        mask_free = map_cleaning == 255                 # free space mask in map_cleaning
        cleaned_acc = np.zeros_like(map_cleaning, dtype=np.uint8)

        # ----------------------------------------------------------------------
        # version 1: only coverage paths
        paths = self.paths.paths_px
        for path in paths:
            # create buffered area around path
            path_points = path.xy - g_xy
            line = LineString(path_points)
            buffered_poly = line.buffer(cleaning_r_px, cap_style=2)
            buffered_poly = np.array(buffered_poly.exterior.coords).astype(int)

            # Draw filled polygon onto image
            cleaned_acc = cv.fillPoly(cleaned_acc, [buffered_poly], color=1)

            # consider only free space pixels
            cleaned_acc = cleaned_acc * mask_free

        # # ----------------------------------------------------------------------
        # # version 2: complete path at once
        # # paths: from sequence paths (full path in pixels)
        # f_path = self.paths.f_path_px

        # # create buffered area around path
        # # cap_style=1 gives flat ends
        # # cap_style=2 gives round ends
        # path_points = f_path.xy - g_xy
        # line = LineString(path_points)
        # buffered_poly = line.buffer(cleaning_r_px, cap_style=2)
        # buffered_poly = np.array(buffered_poly.exterior.coords).astype(int)

        # # Draw filled polygon onto image
        # cleaned_acc = cv.fillPoly(cleaned_acc, [buffered_poly], color=2)

        # consider only free space pixels
        cleaned_acc = cleaned_acc * mask_free
        return cleaned_acc

    def update_map_planning(self, cleaned_acc: np.ndarray) -> None:
        """ update map planning with cleaning paths. """

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
