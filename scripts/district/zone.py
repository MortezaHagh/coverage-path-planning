import os
import json
import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_2_bs_1 import CoveragePP
from scripts.base.configs import DistrictConfgs, PlanningConfigs
# from scripts.planning.coverage_pp_1_simple import CoveragePP


class ZoneStats:
    """Zone statistics class to store zone statistics"""

    def __init__(self):
        self.cleaning_area_px = None
        self.cleaning_area_m = None
        self.cleaned_area_px = None
        self.cleaned_area_m = None
        self.uncleaned_area_m = None
        self.overlaped_area_m = None
        self.coverage_p = None

    def get_data_dict(self) -> dict:
        """Get statistics data as a dictionary."""
        return {
            "cleaning_area_px": int(self.cleaning_area_px),
            "cleaning_area_m": float(self.cleaning_area_m),
            "cleaned_area_px": int(self.cleaned_area_px),
            "cleaned_area_m": float(self.cleaned_area_m),
            "uncleaned_area_m": float(self.uncleaned_area_m),
            "overlaped_area_m": float(self.overlaped_area_m),
            "coverage_p": float(self.coverage_p)
        }

    def print_stats(self):
        print("[Results]: *************************************** ")
        print(f"[Results]: cleaning area: {self.cleaning_area_m:.2f} m^2")
        print(f"[Results]: cleaned area: {self.cleaned_area_m:.2f} m^2")


class ZoneData:
    ind: int
    zmd: ZoneMapData
    dcfg: DistrictConfgs
    pcfgs: PlanningConfigs
    cov_planner: CoveragePP
    st: ZoneStats

    def __init__(self, ind: int = None):
        self.ns = "ZoneData"
        self.lg = MyLogger(self.ns)
        self.ind = ind
        self.zmd = None

        # configs
        self.dcfg = DistrictConfgs()
        self.pcfgs = PlanningConfigs()

        # coverage path planner
        self.cov_planner = None
        self.complete_path = None

        # statistics
        self.st = ZoneStats()

    def set_map_data(self, zmd: ZoneMapData):
        self.zmd = zmd
        self.initialize()

    def initialize(self):
        self.zmd.save_data()
        self.create_map_planning()
        # self.save_map_planning()

    def coverage_planning(self):
        self.cov_planner = CoveragePP(self.zmd)
        self.post_planning()
        # self.save_map_planning()

    # ==================================================================================================================

    def post_planning(self):
        self.calculate_cleaning_area()
        self.add_cleaned_area_to_map()
        self.calculate_cleaned_area()
        self.save_map_planning()
        self.save_map_cleaning_data()
        self.save_result()
        self.st.print_stats()

    def calculate_cleaning_area(self):
        self.st.cleaning_area_px = np.sum(self.zmd.map_cleaning == 255)
        self.st.cleaning_area_m = round(self.st.cleaning_area_px * (self.zmd.reslolution**2), 2)

    def calculate_cleaned_area(self):
        paths_len = self.cov_planner.st.f_path_len_m
        cleaning_r = self.pcfgs.cleaning_r
        area = paths_len * (2 * cleaning_r)
        # overlaped_area_m
        self.st.overlaped_area_m = round(area - self.st.cleaned_area_m, 2)
        # uncleaned_area_m
        self.st.uncleaned_area_m = round(self.st.cleaning_area_m - self.st.cleaned_area_m, 2)
        # coverage percentage
        self.st.coverage_p = round(self.st.cleaned_area_m / self.st.cleaning_area_m * 100, 2)

    def add_cleaned_area_to_map(self):
        map_cleaning = self.zmd.map_cleaning.copy()
        f_path = self.cov_planner.f_path_px
        cleaning_r_px = int(self.pcfgs.cleaning_r / self.zmd.reslolution)

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

        # map_clean_data: show uncleaner area and overlaps
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

        # update map planning with cleaned paths
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
        cleaned_area_m = cleaned_area_px * (self.zmd.reslolution ** 2)
        self.st.cleaned_area_px = cleaned_area_px
        self.st.cleaned_area_m = round(cleaned_area_m, 2)

    # ==================================================================================================================

    def create_map_planning(self):
        map_planning = self.zmd.map_planning
        vertices = self.zmd.vertices
        map_planning = cv.polylines(map_planning, [vertices], isClosed=True, color=(100, 0, 150), thickness=2)
        self.zmd.map_planning = map_planning

    def save_map_planning(self):
        if not self.dcfg.save_map_planning:
            return
        m_name = "map_planning.png"
        map_hl_path = os.path.join(self.zmd.result_path, m_name)
        cv.imwrite(map_hl_path, self.zmd.map_planning)
        # self.lg.info(f"{self.zmd.ind}][save]: map planning saved to: {m_name}")

    def save_map_cleaning_data(self):
        if not self.dcfg.save_map_planning:
            return
        m_name = "map_cleaned.png"
        map_hl_path = os.path.join(self.zmd.result_path, m_name)
        cv.imwrite(map_hl_path, self.zmd.map_clean_data)
        # self.lg.info(f"{self.zmd.ind}][save]: map cleaning saved to: {m_name}")

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        res = self.zmd.reslolution
        return (int(p[0] / res), int(p[1] / res))

    def save_result(self):
        # save zone statistics to json file
        zone_stats = self.st.get_data_dict()
        planner_stats = self.cov_planner.st.get_data_dict()
        result = {
            "zone_ind": self.zmd.ind,
            "method": self.dcfg.method,
            "zone_stats": zone_stats,
            "planner_stats": planner_stats
        }
        # rewrite result file
        with open(self.zmd.result_file, 'wt') as f:
            json.dump(result, f, indent=4)
