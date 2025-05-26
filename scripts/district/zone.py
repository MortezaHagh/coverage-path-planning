import os
import json
import cv2 as cv
import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_2_bs_1 import CoveragePP
from scripts.base.configs import DistrictConfgs, PlanningConfigs
# from scripts.planning.coverage_pp_1_simple import CoveragePP


class ZoneData:
    ind: int
    zmd: ZoneMapData
    cov_planner: CoveragePP
    cleaning_area: float

    def __init__(self, ind: int = None):
        self.ns = "ZoneData"
        self.ind = ind
        self.zmd = None

        # configs
        self.dcfg = DistrictConfgs()
        self.pcfgs = PlanningConfigs()

        # coverage path planner
        self.cov_planner = None

        # statistics - todo
        self.cleaning_area_px = None
        self.cleaning_area_m = None
        self.cleaned_area_px = None
        self.cleaned_area_m = None
        self.uncleaned_area_m = None
        self.overlaped_area_m = None
        self.n_paths = 0

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
        self.n_paths = self.cov_planner.n_paths
        self.calculate_cleaning_area()
        self.add_cleaned_area_to_map()
        self.calculate_cleaned_area()
        self.save_map_planning()
        self.save_map_cleaning_data()
        self.save_result()
        print(f"[{self.ns}-{self.zmd.ind}]: *************************************** ")
        print(f"[{self.ns}-{self.zmd.ind}]: number of paths: {self.n_paths}")
        print(f"[{self.ns}-{self.zmd.ind}]: cleaning area: {self.cleaning_area_m:.2f} m^2")
        print(f"[{self.ns}-{self.zmd.ind}]: cleaned area: {self.cleaned_area_m:.2f} m^2")

    def calculate_cleaning_area(self):
        map_cleaning = self.zmd.map_cleaning
        # calculate the area of the free (white) pixels in the map
        self.cleaning_area_px = np.sum(map_cleaning == 255)
        # convert pixel area to square meters
        self.cleaning_area_m = self.cleaning_area_px * (self.zmd.reslolution**2)
        self.cleaning_area_m = round(self.cleaning_area_m, 2)

    def calculate_cleaned_area(self):
        paths_len = self.cov_planner.total_path_len_m
        cleaning_r = self.pcfgs.cleaning_r
        area = paths_len * (2 * cleaning_r)
        area += np.pi*(cleaning_r**2) * self.n_paths  # add area of the end circles
        # overlaped_area_m
        overlaped_area_m = area - self.cleaned_area_m
        self.overlaped_area_m = round(overlaped_area_m, 2)
        # uncleaned_area_m
        uncleaned_area_m = self.cleaning_area_m - self.cleaned_area_m
        self.uncleaned_area_m = round(uncleaned_area_m, 2)

    def add_cleaned_area_to_map(self):
        map_cleaning = self.zmd.map_cleaning.copy()
        paths = self.cov_planner.paths_px
        cleaning_r_px = int(self.pcfgs.cleaning_r / self.zmd.reslolution)

        # Draw cleaned paths on a mask
        cleaned_acc = np.zeros_like(map_cleaning, dtype=np.uint8)
        hw_bd = self.zmd.hw_bd_px
        h, w = hw_bd["h_min"], hw_bd["w_min"]
        color_p = 1
        for path in paths:
            for i in range(len(path) - 1):
                segment_mask = np.zeros_like(map_cleaning, dtype=np.uint8)
                pt1 = tuple([path[i][0]-w, path[i][1]-h])
                pt2 = tuple([path[i + 1][0]-w, path[i+1][1]-h])
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
        for path in paths:
            for p in path[1:-1]:
                pz = (p[0]-w, p[1]-h)
                map_clean_data = cv.circle(map_clean_data, pz, cleaning_r_px, (0, d_c, 0), -1)
        self.zmd.map_clean_data = map_clean_data

        # update map planning with cleaned paths
        c = 40
        color = (c, c, c)
        map_p_c = self.zmd.map_planning.copy()
        for path in paths:
            for i in range(len(path) - 1):
                pt1 = tuple(map(int, path[i]))
                pt2 = tuple(map(int, path[i + 1]))
                cv.line(map_p_c, pt1, pt2, color=color, thickness=2 * cleaning_r_px)
        alpha = 0.6
        self.zmd.map_planning = cv.addWeighted(self.zmd.map_planning, alpha, map_p_c, 1 - alpha, 0)

        # Only count cleaned area where there was free space (white in map_cleaned)
        # cleaned_area_px = np.sum((cleaned_acc > 0) & (map_cleaned == 255))
        cleaned_area_px = np.sum(cleaned_acc > 0)
        cleaned_area_m = cleaned_area_px * (self.zmd.reslolution ** 2)
        self.cleaned_area_px = cleaned_area_px
        self.cleaned_area_m = round(cleaned_area_m, 2)

    # ==================================================================================================================

    def create_map_planning(self):
        map_planning = self.zmd.map_planning
        vertices = self.zmd.vertices
        map_planning = cv.polylines(map_planning, [vertices], isClosed=True, color=(100, 0, 150), thickness=2)
        self.zmd.map_planning = map_planning

    def save_map_planning(self):
        if not self.dcfg.save_map_planning:
            return
        map_hl_path = self.zmd.path + f"/{self.dcfg.covpp_method}/"
        os.makedirs(map_hl_path, exist_ok=True)
        map_hl_path = map_hl_path + f"map_planning_{self.dcfg.ns}.png"
        cv.imwrite(map_hl_path, self.zmd.map_planning)

    def save_map_cleaning_data(self):
        if not self.dcfg.save_map_planning:
            return
        map_hl_path = self.zmd.path + f"/{self.dcfg.covpp_method}/"
        os.makedirs(map_hl_path, exist_ok=True)
        map_hl_path = map_hl_path + f"clean_data_{self.dcfg.ns}.png"
        cv.imwrite(map_hl_path, self.zmd.map_clean_data)

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        res = self.zmd.reslolution
        return (int(p[0] / res), int(p[1] / res))

    def save_result(self):
        coverage_p = self.cleaned_area_m / self.cleaning_area_m * 100
        coverage_p = round(coverage_p, 2)

        # save zone statistics to json file
        result = {
            "zone_index": self.ind,
            "n_paths": self.n_paths,
            "coverage_percentage": float(coverage_p),
            "total_path_length_m": float(self.cov_planner.total_path_len_m),
            "cleaning_area_m2": float(self.cleaning_area_m),
            "cleaned_area_m2": float(self.cleaned_area_m),
            "uncleaned_area_m2": float(self.uncleaned_area_m),
            "overlaped_area_m2": float(self.overlaped_area_m),
            "cleaning_area_px": float(self.cleaning_area_px),
            "cleaned_area_px": float(self.cleaned_area_px),
        }
        with open(self.zmd.result_file, 'a') as f:
            json.dump(result, f, indent=4)
