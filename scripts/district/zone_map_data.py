import os
import json
import copy
import cv2 as cv
import numpy as np
from scripts.gui.visuals import Rectangle
from scripts.base.map_data import MapData
from scripts.base.my_utils import visualize_map_s
from scripts.base.configs import DistrictConfigs


def convert_to_native(data: dict) -> dict:
    return {k: int(v) if isinstance(v, (np.integer, int))
            else float(round(v, 2)) if isinstance(v, (np.floating, float))
            else v
            for k, v in data.items()}


class ZoneMapData:
    """ZoneMapData class to store zone map data.
        values are in pixel by default.
    """

    def __init__(self, ind: int = 0):
        self.ind: int = ind
        self.name: str = f"Zone_{self.ind}"
        self.zone_path: str = None
        self.maps_path: str = None
        self.result_path: str = None
        self.result_file: str = None
        #
        self.resolution: float = None
        self.orig_map_data: MapData = None
        self.orig_rectangle: Rectangle = None
        self.rectangle: Rectangle = None
        self.final_rectangle: Rectangle = None
        self.rect_data: dict = None                 # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self.vertices: list[tuple[int, int]] = None  # vertices: [(x, y), ...]
        self.hw_bd_px: dict = None                  # for map not rectangle {"h_min": h_min, "w_min": w_min, "h_max": h_max, "w_max": w_max}
        self.final_vertices: np.ndarray = None      # vertices: [(x, y), ...]
        self._final_rect_data: dict = None          # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        self.hw_px: tuple[int, int] = None
        self.map_orig: np.ndarray = None
        self.map_sf_orig: np.ndarray = None
        self.map_cleaning: np.ndarray = None
        #
        self.map_ws: np.ndarray = None
        self.map_segmentation: np.ndarray = None
        self.map_planning: np.ndarray = None

    @property
    def final_rect_data(self) -> int:
        return copy.copy(self._final_rect_data)

    def set_orig_map_data(self, map_data: MapData):
        self.orig_map_data = map_data
        self.resolution = map_data.up_resolution
        self.set_result_path()

    def set_name(self, name: str):
        self.name = name
        self.set_result_path()

    def set_ind(self, ind: int):
        self.ind = ind
        self.name = f"Zone_{self.ind}"
        self.set_result_path()

    def set_result_path(self):
        self.zone_path = os.path.join(self.orig_map_data.bcfg.district_dir, self.name)
        os.makedirs(self.zone_path, exist_ok=True)
        dcfg = DistrictConfigs()
        if self.name == 'District':
            self.result_file = self.zone_path + f"/result_{dcfg.method}.json"
        else:
            self.result_file = self.zone_path + f"/{dcfg.method}/result.json"
            self.result_path = os.path.join(self.zone_path, dcfg.method)
            os.makedirs(self.result_path, exist_ok=True)
        #
        self.maps_path = os.path.join(self.zone_path, "maps")
        os.makedirs(self.maps_path, exist_ok=True)

    def set_orig_rectangle(self, rectangle: Rectangle):
        self.orig_rectangle = rectangle

    def set_rectangle(self, rectangle: Rectangle):
        self.rectangle = rectangle

    def set_rect_data(self, rect_data_t: dict, vertices: list[tuple[int, int]]):
        # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        # vertices: [(x, y), ...]
        self.rect_data = rect_data_t
        self.vertices = vertices

    def set_map_orig(self, map_orig: np.ndarray):
        self.map_orig = map_orig

    def set_map_ws(self, map_ws: np.ndarray):
        self.map_ws = map_ws

    def set_map_sf_orig(self, map_sf_orig: np.ndarray):
        self.map_sf_orig = map_sf_orig

    def set_map_cleaning(self, map_cleaning: np.ndarray):
        self.map_cleaning = map_cleaning
        self.hw_px = map_cleaning.shape

    def set_map_viz(self, map_viz: np.ndarray):
        self.map_planning = map_viz

    def set_hw_bd_px(self, hw_bd_px: dict):
        # cropped hw limits
        self.hw_bd_px = hw_bd_px

    def set_final_rect_data(self, rect_data_t: dict, vertices: list[tuple[int, int]]):
        # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        # vertices: [(x, y), ...]
        self._final_rect_data = rect_data_t
        self.final_vertices = vertices
        self.final_rectangle = Rectangle(self.final_rect_data)

    def save_data(self) -> None:
        """ save zone map data to files.
            maps, local_map.json.
        """

        #
        cv.imwrite(self.maps_path + "/" + "local_map_1_orig.png", self.map_orig)
        cv.imwrite(self.maps_path + "/" + "local_map_2_sf_orig.png", self.map_sf_orig)
        cv.imwrite(self.maps_path + "/" + "local_map_3_cleaning.png", self.map_cleaning)
        cv.imwrite(self.maps_path + "/" + "local_map_4_viz.png", self.map_planning)
        #
        zone_data_d = dict()
        zone_data_d["origin_m"] = self.orig_map_data.origin_m
        zone_data_d["up_resolution"] = self.orig_map_data.up_resolution
        zone_data_d["hw_bd_px"] = self.hw_bd_px
        zone_data_d["rect_data"] = convert_to_native(self.rect_data)
        zone_data_d["final_rect_data"] = convert_to_native(self.final_rect_data)

        with open(self.maps_path + "/" + "local_map.json", "w") as f:
            json_object = json.dumps(zone_data_d, indent=4)
            f.write(json_object)

    def visualize(self, case: str = "cleaning"):
        if case == "cleaning":
            visualize_map_s(self.map_cleaning, f"zone_{self.ind}_{case}")
        elif case == "viz":
            visualize_map_s(self.map_viz, f"zone_{self.ind}_{case}")
        else:
            visualize_map_s(self.map_orig, f"zone_{self.ind}_{'orig'}")
