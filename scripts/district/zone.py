import os
import cv2 as cv
import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_2_bs_1 import CoveragePP
from scripts.base.configs import SpaceConfigs, DistrictConfgs
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
        self.scfg = SpaceConfigs()
        self.dcfg = DistrictConfgs()

        # coverage path planner
        self.cov_planner = None

        # statistics - todo
        self.cleaning_area_px = None
        self.cleaning_area_m = None
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
        self.save_map_planning()

    # ==================================================================================================================

    def post_planning(self):
        self.calculate_cleaning_area()
        self.n_paths = self.cov_planner.n_paths
        print(f"[{self.ns}-{self.zone_map_data.ind}]: *************************************** ")
        print(f"[{self.ns}-{self.zone_map_data.ind}]: cleaning area: {self.cleaning_area_m:.2f} m^2")
        print(f"[{self.ns}-{self.zone_map_data.ind}]: number of paths: {self.n_paths}")

    def calculate_cleaning_area(self):
        map_cleaning = self.zone_map_data.map_cleaning
        # calculate the area of the free (white) pixels in the map
        self.cleaning_area_px = np.sum(map_cleaning == 255)
        # convert pixel area to square meters
        self.cleaning_area_m = self.cleaning_area_px * (self.zone_map_data.reslolution**2)

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
