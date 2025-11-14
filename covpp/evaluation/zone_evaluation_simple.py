import cv2 as cv
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.base_utils import MyPath
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.sequence_handler import SequencePaths
from covpp.evaluation.zone_evaluation_base import ZoneStatsBase, ZoneEvaluationBase


class ZoneEvaluationSimple(ZoneEvaluationBase):
    """Zone evaluation class to evaluate zone statistics.
    """

    def __init__(self, zmd: ZoneMapData, paths: SequencePaths):
        super().__init__(zmd, paths)
        self.lg = MyLogger("ZoneEvaluationSimple")

    def evaluate(self, times: dict = None):
        self.stats = ZoneStatsBase()
        self.stats.map_name = self.zmd.orig_map_data.name
        self.stats.method = self.dcfg.method
        self.stats.ind = self.zmd.ind

        if not self.check_paths_validity():
            self.save_result()
            return

        self.calculate_time_statistics(times)
        self.calculate_path_statistics()
        self.calculate_travel_times()
        #
        cleaned_acc = self.calculate_cleaned_accumulated_area()
        self.calculate_cleaning_area()
        self.calculate_cleaned_area_stats(cleaned_acc)
        self.update_map_planning(cleaned_acc)

        self.save_result()

    def check_paths_validity(self) -> bool:
        """ check if paths are valid for evaluation. """
        # paths: from sequence paths (full path in pixels)
        if self.paths.f_path_px is None:
            self.lg.warn("fill path is None. Cannot add cleaned area to map.")
            self.zmd.map_cleaned = self.zmd.map_cleaning.copy()
            return False
        return True

    def calculate_time_statistics(self, times: dict) -> None:
        if times is None:
            return
        self.stats.time_segmentation = times['segmentation'] - times['start']
        self.stats.time_coverage_planning = times['coverage_planning'] - times['segmentation']
        self.stats.time_sequencing = times['sequencing'] - times['coverage_planning']
        self.stats.time_total = times['sequencing'] - times['start']

    def calculate_path_statistics(self) -> None:
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
        t_lin = self.stats.full_length / self.robot_cfgs.v_lin_avg
        t_ang = self.stats.full_yaw_rad / self.robot_cfgs.v_ang_avg
        self.stats.full_trav_time = t_lin + t_ang
        #
        t_lin = self.stats.coverage_length / self.robot_cfgs.v_lin_avg
        t_ang = self.stats.coverage_yaw_rad / self.robot_cfgs.v_ang_avg
        self.stats.coverage_trav_time = t_lin + t_ang

    # ------------------------------------------------------------------------------------------------------------------
    # Cleaning area calculations
    # ------------------------------------------------------------------------------------------------------------------

    def calculate_cleaning_area(self):
        self.stats.cleaning_area_px = np.sum(self.zmd.map_cleaning == 255)
        self.stats.cleaning_area_m = self.stats.cleaning_area_px * (self.zmd.resolution**2)

    def calculate_cleaned_area_stats(self, cleaned_acc: np.ndarray) -> None:

        cleaned_area_px = np.sum(cleaned_acc > 0)
        cleaned_area_m = cleaned_area_px * (self.zmd.resolution ** 2)
        self.stats.cleaned_area_px = cleaned_area_px
        self.stats.cleaned_area_m = cleaned_area_m

        # overlaped_area
        overlapped_area_px = np.sum(cleaned_acc > 1)
        self.stats.overlaped_area_m = overlapped_area_px * (self.zmd.resolution ** 2)
        self.stats.overlap_p = self.stats.overlaped_area_m / self.stats.cleaning_area_m * 100

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
        mask_free = (map_cleaning == 255)               # free space mask in map_cleaning

        # paths: from sequence paths (full path in pixels)
        f_path = self.paths.f_path_px

        cleaned_acc = np.zeros_like(map_cleaning, dtype=np.uint8)
        color_p = 1
        # for path in paths: draw cleaned path segments on cleaned_acc
        for i in range(f_path.n - 1):
            segment_mask = np.zeros_like(map_cleaning, dtype=np.uint8)
            pt1 = tuple(map(int, [f_path.point(i).x-g_xy[0], f_path.point(i).y-g_xy[1]]))
            pt2 = tuple(map(int, [f_path.point(i+1).x-g_xy[0], f_path.point(i+1).y-g_xy[1]]))
            cv.line(segment_mask, pt1, pt2, color=color_p, thickness=2 * cleaning_r_px)
            cleaned_acc = cv.add(cleaned_acc, segment_mask)
        # consider only free space pixels
        cleaned_acc = cleaned_acc * mask_free
        return cleaned_acc

    def update_map_planning(self, cleaned_acc: np.ndarray) -> None:
        """ update map planning with cleaning paths. """

        # [map_cleaned]: show uncleaned area and overlaps, only for visualization
        # uncleaned free pixels -> red, cleaned -> scaled green

        # data
        cleaning_r_px = int(self.pcfg.cleaning_r / self.zmd.resolution)  # cleaning radius in pixels
        g_xy = self.zmd.global_xy  # global offset

        #
        mask_free = self.zmd.map_cleaning == 255
        max_overlap = np.max(cleaned_acc)  # maximum overlap count among pixels

        d_c = int(np.floor(250 / max_overlap))
        map_cleaned = cv.cvtColor(self.zmd.map_cleaning.copy(), cv.COLOR_GRAY2BGR)
        mask_zero = mask_free & (cleaned_acc == 0)
        mask_nonzero = mask_free & (cleaned_acc > 0)
        map_cleaned[mask_zero] = (0, 0, 200)

        if np.any(mask_nonzero):
            green = np.clip(d_c * cleaned_acc, 0, 255).astype(np.uint8)
            map_cleaned[..., 1][mask_nonzero] = green[mask_nonzero]
        # for path in paths: draw circles at path points
        for p in self.paths.f_path_px.points[1:-1]:
            pz = (p.x-g_xy[0], p.y-g_xy[1])
            map_cleaned = cv.circle(map_cleaned, pz, cleaning_r_px, (0, d_c, 0), -1)
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
