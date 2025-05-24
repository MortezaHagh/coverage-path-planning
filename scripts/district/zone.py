import os
import cv2 as cv
import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_2_bs_1 import CoveragePP
from scripts.base.configs import DistrictConfgs, PlanningConfigs
# from scripts.planning.coverage_pp_1_simple import CoveragePP


class ZoneData:
    ind: int
    zone_map_data: ZoneMapData
    cov_planner: CoveragePP
    cleaning_area: float

    def __init__(self, ind: int = None):
        self.ns = "ZoneData"
        self.ind = ind
        self.zone_map_data = None

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
        self.n_paths = 0

    def set_map_data(self, zone_map_data: ZoneMapData):
        self.zone_map_data = zone_map_data
        self.initialize()

    def initialize(self):
        self.zone_map_data.save_data()
        self.create_map_planning()
        # self.save_map_planning()

    def coverage_planning(self):
        self.cov_planner = CoveragePP(self.zone_map_data)
        self.post_planning()
        # self.save_map_planning()

    # ==================================================================================================================

    def post_planning(self):
        self.calculate_cleaning_area()
        self.calculate_cleaned_area()
        self.save_map_planning()
        self.n_paths = self.cov_planner.n_paths
        print(f"[{self.ns}-{self.zone_map_data.ind}]: *************************************** ")
        print(f"[{self.ns}-{self.zone_map_data.ind}]: cleaning area: {self.cleaning_area_m:.2f} m^2")
        print(f"[{self.ns}-{self.zone_map_data.ind}]: number of paths: {self.n_paths}")
        print(f"[{self.ns}-{self.zone_map_data.ind}]: cleaned area: {self.cleaned_area_m:.2f} m^2")

    def calculate_cleaning_area(self):
        map_cleaning = self.zone_map_data.map_cleaning
        # calculate the area of the free (white) pixels in the map
        self.cleaning_area_px = np.sum(map_cleaning == 255)
        # convert pixel area to square meters
        self.cleaning_area_m = self.cleaning_area_px * (self.zone_map_data.reslolution**2)

    def calculate_cleaned_area(self):
        map_cleaned = self.zone_map_data.map_cleaning.copy()
        paths = self.cov_planner.paths_px
        cleaning_r_px = int(self.pcfgs.cleaning_r / self.zone_map_data.reslolution)

        # Draw cleaned paths on a mask
        cleaned_acc = np.zeros_like(map_cleaned, dtype=np.uint8)
        hw_bd = self.zone_map_data.hw_bd_px
        h, w = hw_bd["h_min"], hw_bd["w_min"]
        color_p = 40
        for path in paths:
            segment_mask = np.zeros_like(map_cleaned, dtype=np.uint8)
            for i in range(len(path) - 1):
                pt1 = tuple([path[i][0]-w, path[i][1]-h])
                pt2 = tuple([path[i + 1][0]-w, path[i+1][1]-h])
                cv.line(segment_mask, pt1, pt2, color=color_p, thickness=2 * cleaning_r_px)
            cleaned_acc = cv.add(cleaned_acc, segment_mask)

        # Optionally, mark the cleaned area in a different color for visualization
        c = 40
        color = (c, c, c)
        map_cleaned_viz = self.zone_map_data.map_planning.copy()
        for path in paths:
            for i in range(len(path) - 1):
                pt1 = tuple(map(int, path[i]))
                pt2 = tuple(map(int, path[i + 1]))
                cv.line(map_cleaned_viz, pt1, pt2, color=color, thickness=2 * cleaning_r_px)
        alpha = 0.5
        self.zone_map_data.map_planning = cv.addWeighted(self.zone_map_data.map_planning, alpha, map_cleaned_viz, 1 - alpha, 0)

        # Only count cleaned area where there was free space (white in map_cleaned)
        cleaned_area_px = np.sum((cleaned_acc > 0) & (map_cleaned == 255))
        # Convert to square meters
        cleaned_area_m = cleaned_area_px * (self.zone_map_data.reslolution ** 2)

        self.cleaned_area_px = cleaned_area_px
        self.cleaned_area_m = cleaned_area_m

    # ==================================================================================================================

    def create_map_planning(self):
        map_planning = self.zone_map_data.map_viz.copy()
        vertices = self.zone_map_data.vertices
        map_planning = cv.polylines(map_planning, [vertices], isClosed=True, color=(100, 0, 150), thickness=2)
        self.zone_map_data.map_planning = map_planning

    def save_map_planning(self):
        if not self.dcfg.save_map_planning:
            return
        map_hl_path = self.zone_map_data.path + f"/{self.dcfg.covpp_method}/"
        os.makedirs(map_hl_path, exist_ok=True)
        map_hl_path = map_hl_path + f"map_planning_{self.dcfg.ns}.png"
        cv.imwrite(map_hl_path, self.zone_map_data.map_planning)

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        res = self.zone_map_data.reslolution
        return (int(p[0] / res), int(p[1] / res))
