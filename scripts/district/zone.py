import os
import json
import copy
import cv2 as cv
import numpy as np
from scripts.gui.visuals import Rectangle
from scripts.base.map_data import MapData
from scripts.base.my_utils import visualize_map_s


class ZoneMapData:
    ind: int
    map_data: MapData
    rectangle: Rectangle
    map_orig: np.ndarray
    map_sf_orig: np.ndarray
    map_cleaning: np.ndarray
    map_ws: np.ndarray
    map_viz: np.ndarray
    hw: tuple[int, int]
    bb_px: dict
    vertices: np.ndarray  # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
    _rect_data: dict  # vertices: [(x, y), ...]

    def __init__(self, ind: int = 0):
        self.ind = ind
        self.map_data = None
        self.rectangle = None
        self.map_orig = None
        self.map_sf_orig = None
        self.map_cleaning = None
        self.map_ws = None
        self.map_viz = None
        self.hw = None
        self.bb_px = None
        self.vertices = None
        self._rect_data = None

    @property
    def rect_data(self) -> int:
        return copy.copy(self._rect_data)

    def set_ind(self, ind: int):
        self.ind = ind

    def set_map_data(self, map_data: MapData):
        self.map_data = map_data

    def set_rectangle(self, rectangle: Rectangle):
        self.rectangle = rectangle

    def set_map_orig(self, map_orig: np.ndarray):
        self.map_orig = map_orig

    def set_map_ws(self, map_ws: np.ndarray):
        self.map_ws = map_ws

    def set_map_sf_orig(self, map_sf_orig: np.ndarray):
        self.map_sf_orig = map_sf_orig

    def set_map_cleaning(self, map_cleaning: np.ndarray):
        self.map_cleaning = map_cleaning
        self.hw = map_cleaning.shape

    def set_map_viz(self, map_viz: np.ndarray):
        self.map_viz = map_viz

    def set_bb_px(self, bb_px: dict):
        self.bb_px = bb_px

    def set_vertices(self, vertices: np.ndarray):
        # vertices: [(x, y), ...]
        self.vertices = vertices

    def set_rect_data(self, rect_data_t: dict):
        # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self._rect_data = rect_data_t

    # todo
    # def make_frame(self, value: int = 0):
    #     """ make frame of the map_cleaning as value (default = 0) """
    #     self.map_cleaning[:, 0] = value
    #     self.map_cleaning[:, -1] = value
    #     self.map_cleaning[0, :] = value
    #     self.map_cleaning[-1, :] = value

    def save_data(self, path: str) -> None:
        res_path = os.path.join(self.map_data.bcfg.map_path, path)
        os.makedirs(res_path, exist_ok=True)
        #
        cv.imwrite(res_path + "/" + "local_map_1_orig.png", self.map_orig)
        cv.imwrite(res_path + "/" + "local_map_2_sf_orig.png", self.map_sf_orig)
        cv.imwrite(res_path + "/" + "local_map_3_cleaning.png", self.map_cleaning)
        cv.imwrite(res_path + "/" + "local_map_4_viz.png", self.map_viz)
        #
        zone_map_data = {"coords_px": self.bb_px}
        zone_map_data["up_resolution"] = self.map_data.up_resolution
        zone_map_data["origin_m"] = self.map_data.origin_m
        with open(res_path + "/" + "local_map.json", "w") as f:
            json_object = json.dumps(zone_map_data, indent=4)
            f.write(json_object)

    def visualize(self, case: str = "cleaning"):
        if case == "cleaning":
            visualize_map_s(self.map_cleaning, f"zone_{self.ind}_{case}")
        elif case == "viz":
            visualize_map_s(self.map_viz, f"zone_{self.ind}_{case}")
        else:
            visualize_map_s(self.map_orig, f"zone_{self.ind}_{'orig'}")
