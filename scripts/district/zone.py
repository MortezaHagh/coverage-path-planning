import numpy as np
import cv2 as cv
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import SpaceConfigs


class ZoneData:
    ind: int
    zone_map_data: ZoneMapData
    map_hl: np.ndarray
    os_path_m: list[tuple[float, float]]
    cleaning_area: float

    def __init__(self, ind: int = None):
        self.ind = ind
        self.zone_map_data = None
        self.map_hl = None
        self.os_path_m = []

        #
        self.scfg = SpaceConfigs()

        # statistics - todo
        self.cleaning_area = None

    def create_map_hl(self):
        """create map high level"""
        map_hl = self.zone_map_data.map_viz.copy()
        vertices = self.zone_map_data.vertices
        self.map_hl = cv.polylines(map_hl, [vertices], isClosed=True, color=(100, 0, 150), thickness=4)
        self.save_map_hl()

    def save_map_hl(self):
        base_path = self.zone_map_data.orig_map_data.bcfg.district_dir
        map_hl_path = base_path + f"/Zone_{self.ind}"
        cv.imwrite(map_hl_path + "/" + "map_hl.png", self.map_hl)

    def add_os_path_map_hl(self):
        path = self.os_path_m
        for (p1, p2) in zip(path[:-1], path[1:]):
            p1 = self.point_m_to_px(p1)
            p2 = self.point_m_to_px(p2)
            self.map_hl = cv.line(self.map_hl, p1, p2, (0, 255, 10), 2)
        #
        # vis robot
        alpha = 0.1
        pose = self.point_m_to_px(path[0])
        radius_px_1 = int(np.ceil(self.scfg.ws_r_dist / self.zone_map_data.orig_map_data.up_resolution))
        radius_px_2 = int(np.ceil(self.scfg.robot_radius / self.zone_map_data.orig_map_data.up_resolution))
        radius_px_3 = int(np.ceil(self.scfg.cleaning_r / self.zone_map_data.orig_map_data.up_resolution))
        overlay = self.map_hl.copy()
        cv.circle(overlay, pose, radius_px_1, [0, 250, 250], -1)
        cv.circle(overlay, pose, radius_px_2, [0, 0, 250], -1)
        cv.circle(overlay, pose, radius_px_3, [250, 100, 0], -1)
        cv.circle(overlay, pose, 5, [250, 250, 250], -1)
        self.map_hl = cv.addWeighted(overlay, alpha, self.map_hl, 1 - alpha, 0)
        #
        self.save_map_hl()

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        res = self.zone_map_data.reslolution
        return (int(p[0] / res), int(p[1] / res))
