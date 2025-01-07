import os
import copy
import cv2 as cv
import numpy as np
from scripts.map_data import MapData
from scripts.configs import BConfigs
from scripts.gui.visuals import Rectangle
from scripts.my_utils import visualize_map


class SpaceAnalyzer:
    map_data: MapData
    binary_map: np.ndarray
    vertices: np.ndarray
    # district_rect: Rectangle

    def __init__(self, map_data: MapData, configs: BConfigs, district_rect: Rectangle):
        self.ns = "SpaceAnalyzer"

        # analyzed space path
        self.res_path = os.path.join(configs.map_path, configs.district_folder)
        os.makedirs(self.res_path, exist_ok=True)

        # inputs
        self.binary_map = map_data.up_bin_map     # binary map
        self.map_data = map_data
        self.district_rect = district_rect
        self.vertices = np.array(district_rect.vertices_t, dtype=np.int32)  # district vertices

        # attributes
        self.corners: dict[str:int] = None

        #
        self.select_space()

        # #
        # zone = MapInfo(free_space_map, available_map, zone_coordinates, mapPgm.shape, mapOrigin)

    def select_space(self):
        # print(f"[{self.ns}]: vertices:", self.vertices)

        # crop map
        # croppedMap, outsideOfZoneMask =
        self.crop_map()

    def crop_map(self):
        # check bounds
        wh_max = np.max(self.vertices, axis=0)
        wh_min = np.min(self.vertices, axis=0)
        w_max, h_max = wh_max.tolist()
        w_min, h_min = wh_min.tolist()
        w_min = max(0, w_min)
        h_min = max(0, h_min)
        w_max = min(w_max, self.map_data.updated_w-1)
        h_max = min(h_max, self.map_data.updated_h-1)

        # corners
        # self.cleaningZoneBBox = [h_min, h_max, w_min, w_max]
        self.corners = {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        # print(f"[{self.ns}]: corners: ", self.corners)

        # cropped map and vertices
        cropped_map = copy.copy(self.binary_map[h_min:h_max, w_min:w_max])
        cropped_vertices = np.zeros_like(self.vertices)
        for i, v in enumerate(self.vertices):
            cropped_vertices[i] = np.array([v[0] - w_min, v[1] - h_min])
        # visualize_map(cropped_map)

        # outside mask
        out_mask = np.ones_like(cropped_map) * 255

        # # outside mask
        # out_mask = np.zeros_like(cropped_map)
        # # fill mask with white pixels
        # cv.fillConvexPoly(out_mask, cropped_vertices, 255)
        # # replace pixel outside of the zone with black pixels
        # for hp in range(0, h_max - h_min):
        #     for wp in range(0, w_max - w_min):
        #         if out_mask[hp, wp] == 0:
        #             cropped_map[hp, wp] = 0
        #             # never happens!

        #
        visualize_map(cropped_map)
        visualize_map(out_mask)

        # return croppedMap, outsideOfZoneMask
