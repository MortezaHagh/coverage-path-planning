import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.sequence.sequence import Sequence
from scripts.evaluation.zone_evaluation_base import ZoneStatsBase, ZoneEvaluationBase


class ZoneEvaluationSimple(ZoneEvaluationBase):
    """Zone evaluation Base class to evaluate zone statistics.
    """

    def __init__(self, sequence: Sequence):
        super().__init__(sequence)
        self.lg = MyLogger("ZoneEvaluationSimple")

    def evaluate(self, duration: float = 0.0):
        self.stats = ZoneStatsBase()
        self.stats.processing_time = float(duration, 3)
        #
        self.calculate_cleaning_area()
        self.add_cleaned_area_to_map()
        self.calculate_cleaned_area()
        self.save_result()

    def calculate_cleaning_area(self):
        self.stats.cleaning_area_px = np.sum(self.zmd.map_cleaning == 255)
        self.stats.cleaning_area_m = round(self.stats.cleaning_area_px * (self.zmd.resolution**2), 2)

    def calculate_cleaned_area(self):
        paths_len = self.sequence.st.len_m
        cleaning_r = self.pcfgs.cleaning_r
        area = paths_len * (2 * cleaning_r)
        # overlaped_area_m
        self.stats.overlaped_area_m = round(area - self.stats.cleaned_area_m, 2)
        self.stats.overlap_p = round(self.stats.overlaped_area_m / self.stats.cleaning_area_m * 100, 2)
        # uncleaned_area_m
        self.stats.uncleaned_area_m = round(self.stats.cleaning_area_m - self.stats.cleaned_area_m, 2)
        # coverage percentage
        self.stats.coverage_p = round(self.stats.cleaned_area_m / self.stats.cleaning_area_m * 100, 2)

    def add_cleaned_area_to_map(self):
        """ add cleaned area to map_clean_data and update map_planning with cleaning paths. """

        # map_cleaning: from zone map data (without any paths)
        # used for creating map_clean_data
        map_cleaning = self.zmd.map_cleaning.copy()
        f_path = self.sequence.f_path_px
        cleaning_r_px = int(self.pcfgs.cleaning_r / self.zmd.resolution)

        if f_path is None:
            self.lg.error("fill path is None. Cannot add cleaned area to map.")
            self.zmd.map_clean_data = map_cleaning.copy()
            return

        # Draw cleaned paths on a mask
        cleaned_acc = np.zeros_like(map_cleaning, dtype=np.uint8)
        hw_bd = self.zmd.hw_bd_px
        h, w = hw_bd["h_min"], hw_bd["w_min"]
        color_p = 1
        # for path in paths:
        for i in range(f_path.n - 1):
            segment_mask = np.zeros_like(map_cleaning, dtype=np.uint8)
            pt1 = tuple(map(int, [f_path.point(i).x-w, f_path.point(i).y-h]))
            pt2 = tuple(map(int, [f_path.point(i+1).x-w, f_path.point(i+1).y-h]))
            cv.line(segment_mask, pt1, pt2, color=color_p, thickness=2 * cleaning_r_px)
            cleaned_acc = cv.add(cleaned_acc, segment_mask)
        max_overlap = np.max(cleaned_acc)

        # map_clean_data: show uncleaned area and overlaps
        d_c = int(np.floor(250/max_overlap))
        map_clean_data = map_cleaning.copy()
        map_clean_data = cv.cvtColor(map_clean_data, cv.COLOR_GRAY2BGR)
        n, m = map_clean_data.shape[:2]
        for i in range(n):
            for j in range(m):
                if map_cleaning[i, j] == 255:
                    if cleaned_acc[i, j] == 0:
                        map_clean_data[i, j] = (0, 0, 200)  # bgr
                    else:
                        map_clean_data[i, j] = (0, d_c*cleaned_acc[i, j], 0)
        # for path in paths:
        for p in f_path.points[1:-1]:
            pz = (int(p.x-w), int(p.y-h))
            map_clean_data = cv.circle(map_clean_data, pz, cleaning_r_px, (0, d_c, 0), -1)
        self.zmd.map_clean_data = map_clean_data

        # update map planning with cleaning paths
        add_cleaned_area_to_map_planning = False
        if add_cleaned_area_to_map_planning:
            c = 40
            color = (c, c, c)
            map_p_c = self.zmd.map_planning.copy()
            # for path in paths:
            for i in range(f_path.n - 1):
                pt1 = tuple(map(int, f_path.point(i).xy))
                pt2 = tuple(map(int, f_path.point(i+1).xy))
                cv.line(map_p_c, pt1, pt2, color=color, thickness=2 * cleaning_r_px)
            alpha = 0.6
            self.zmd.map_planning = cv.addWeighted(self.zmd.map_planning, alpha, map_p_c, 1 - alpha, 0)

        # Only count cleaned area where there was free space (white in map_cleaned)
        # cleaned_area_px = np.sum((cleaned_acc > 0) & (map_cleaned == 255))
        cleaned_area_px = np.sum(cleaned_acc > 0)
        cleaned_area_m = cleaned_area_px * (self.zmd.resolution ** 2)
        self.stats.cleaned_area_px = cleaned_area_px
        self.stats.cleaned_area_m = round(cleaned_area_m, 2)
