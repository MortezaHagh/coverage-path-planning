import os
import cv2 as cv
import numpy as np
from scripts.map_data import MapData
from scripts.configs import BConfigs
from scripts.gui.visuals import Rectangle


class SpaceAnalyzer:
    binary_map: np.ndarray

    def __init__(self, map_data: MapData, configs: BConfigs, district_rect: Rectangle):

        # analyzed space path
        self.res_path = os.path.join(configs.map_path, configs.district_folder)
        os.makedirs(self.res_path, exist_ok=True)

        # inputs
        self.binary_map = map_data.up_bin_map
        self.rect = district_rect

        # #
        # select_space(vertices=rectangle.get_vertices())

        # #
        # zone = MapInfo(free_space_map, available_map, zone_coordinates, mapPgm.shape, mapOrigin)

    def select_space(self):
        pass
