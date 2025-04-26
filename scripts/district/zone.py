import cv2 as cv
import numpy as np
from scripts.base.configs import SpaceConfigs
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_2_bs_1 import CoveragePP
# from scripts.planning.coverage_pp_1_simple import CoveragePP


class ZoneData:
    ind: int
    zone_map_data: ZoneMapData
    cov_planner: CoveragePP
    cleaning_area: float

    def __init__(self, ind: int = None):
        self.ind = ind
        self.zone_map_data = None

        # configs
        self.scfg = SpaceConfigs()

        # coverage path planner
        self.cov_planner = None

        # statistics - todo
        self.cleaning_area = None

    def set_map_data(self, zone_map_data: ZoneMapData):
        self.zone_map_data = zone_map_data
        self.initialize()

    def initialize(self):
        # self.zone_map_data.save_data() # todo
        self.create_map_planning()
        # self.save_map_planning()

    def coverage_planning(self):
        self.cov_planner = CoveragePP(self.zone_map_data)
        # self.add_os_path_to_map_planning()
        self.save_map_planning()

    # ==================================================================================================================
    # ==================================================================================================================

    def create_map_planning(self):
        """create map planning"""
        map_planning = self.zone_map_data.map_viz.copy()
        vertices = self.zone_map_data.vertices
        self.zone_map_data.map_planning = cv.polylines(map_planning, [vertices], isClosed=True, color=(100, 0, 150), thickness=4)

    def save_map_planning(self):
        base_path = self.zone_map_data.orig_map_data.bcfg.district_dir
        map_hl_path = base_path + f"/Zone_{self.ind}"
        cv.imwrite(map_hl_path + "/" + "map_planning.png", self.zone_map_data.map_planning)

    def add_os_path_to_map_planning(self):
        path = self.cov_planner.path_px
        for (p1, p2) in zip(path[:-1], path[1:]):
            self.zone_map_data.map_planning = cv.line(self.zone_map_data.map_planning, p1, p2, (0, 255, 10), 2)

        # vis robot
        alpha = 0.2
        pose = self.point_m_to_px(path[0])
        radius_px_1 = int(np.ceil(self.scfg.ws_r_dist / self.zone_map_data.orig_map_data.up_resolution))
        radius_px_2 = int(np.ceil(self.scfg.robot_radius / self.zone_map_data.orig_map_data.up_resolution))
        radius_px_3 = int(np.ceil(self.scfg.cleaning_r / self.zone_map_data.orig_map_data.up_resolution))
        overlay = self.zone_map_data.map_planning.copy()
        cv.circle(overlay, pose, radius_px_1, [200, 200, 250], -1)
        cv.circle(overlay, pose, radius_px_2, [0, 0, 250], -1)
        cv.circle(overlay, pose, radius_px_3, [100, 0, 100], -1)
        cv.circle(overlay, pose, 5, [250, 250, 250], -1)
        self.zone_map_data.map_planning = cv.addWeighted(overlay, alpha, self.zone_map_data.map_planning, 1 - alpha, 0)
        #
        self.save_map_planning()

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        res = self.zone_map_data.reslolution
        return (int(p[0] / res), int(p[1] / res))
