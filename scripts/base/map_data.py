""" MapData data classes """
import numpy as np
import cv2 as cv
from scripts.base.configs import BConfigs


class MapData:
    """ original and updated map data """
    bcfg: BConfigs  # base configs
    name: str  # map name
    orig_h: int  # original height
    orig_w: int  # original width
    orig_c_w: int  # original center width
    orig_c_h: int  # original center height
    updated_h: int  # updated height
    updated_w: int  # updated width
    updated_wh: tuple[int, int]  # updated width and height tuple
    up_c_w: int  # updated center width
    up_c_h: int  # updated center height
    origin_m: tuple[int, int]  # origin in meters
    #
    orig_map: np.ndarray  # original grayscale map
    orig_bin_map: np.ndarray  # original binary map
    up_map: np.ndarray  # updated grayscale map
    up_rgb_map: np.ndarray  # updated rgb map
    up_bin_map: np.ndarray  # updated binary map
    #
    resolution: float  # original resolution
    up_resolution: float  # updated resolution

    def __init__(self):
        self.ns = "MapData"
        #
        self.bcfg = None
        self.name = None
        self.orig_h = None
        self.orig_w = None
        self.orig_c_w = None
        self.orig_c_h = None
        self.updated_h = None
        self.updated_w = None
        self.updated_wh = None
        self.up_c_w = None
        self.up_c_h = None
        self.origin_m = None
        #
        self.wind_hw = (1080, 1920)
        self.wind_h = None
        self.wind_w = None
        self.wind_center = None
        self.zoom = None
        #
        self.orig_map = None
        self.orig_bin_map = None
        self.up_map = None
        self.up_rgb_map = None
        self.up_bin_map = None
        #
        self.resolution = None
        self.up_resolution = None

    def set_basic_configs(self, bcfg: BConfigs, name: str = None):
        if name:
            name = self.name
        self.bcfg = bcfg

    def set_origin_m(self, origin_m: tuple[float, float]):
        self.origin_m = origin_m

    def set_resolution(self, res):
        self.resolution = res
        self.up_resolution = round(self.resolution / self.zoom, 4)

    def set_bin_map(self, bin_map):
        self.orig_bin_map = bin_map
        self.up_bin_map = cv.resize(self.orig_bin_map, self.updated_wh, interpolation=cv.INTER_AREA)

    def set_orig_map(self, map_data: np.ndarray):
        self.orig_map = map_data
        #
        self.set_map_hw()
        self.set_wind_wh()
        self.resize_map()
        self.create_rgb_map()
        self.set_orig_center()
        self.set_updated_center()

    def set_map_hw(self):
        height, width = self.orig_map.shape[:2]
        self.orig_h = height
        self.orig_w = width
        print(f"[{self.ns}]: Map size [w,h]: {width}x{height}")

    def set_wind_wh(self, wind_hw=None):
        if wind_hw is None:
            wind_hw = self.wind_hw
        self.wind_h, self.wind_w = self.wind_hw
        #
        zoom = min(self.wind_h/self.orig_h, self.wind_w/self.orig_w)
        updated_h = int(self.orig_h * zoom)
        updated_w = int(self.orig_w * zoom)
        self.zoom = zoom
        self.set_updated_map_wh(h=updated_h, w=updated_w)

    def set_updated_map_wh(self, w, h):
        self.updated_h = h
        self.updated_w = w
        self.updated_wh = (w, h)
        print(f"[{self.ns}]: Updated Map size [w,h]: {w}x{h}")

    def resize_map(self):
        self.up_map = cv.resize(self.orig_map, self.updated_wh, interpolation=cv.INTER_AREA)

    def create_rgb_map(self):
        self.up_rgb_map = cv.cvtColor(self.up_map, cv.COLOR_GRAY2BGR)

    def set_orig_center(self):
        self.orig_c_w = self.orig_w//2
        self.orig_c_h = self.orig_h//2

    def set_updated_center(self):
        self.up_c_w = self.updated_w//2
        self.up_c_h = self.updated_h//2

    def point_to_map(self, x, y):
        mx = (x - self.wind_w // 2) + self.up_c_w
        my = (y - self.wind_h // 2) + self.up_c_h
        # mx = (x - self.wind_w // 2) / self.zoom + self.orig_c_w
        # my = (y - self.wind_h // 2) / self.zoom + self.orig_c_h
        return (mx, my)

    def point_to_up_map(self, x, y):
        mx = (x - self.up_c_w // 2) + self.orig_c_w
        my = (y - self.up_c_h // 2) + self.orig_c_h
        return (mx, my)

    def convert_map_to_world(self, p_m: tuple[int, int]) -> tuple[float, float]:  # todo todel
        """[x, y] [w, h] from map (pixels) to world (meters)"""
        p_w_0 = p_m[0] * self.up_resolution + self.orig_w
        p_w_1 = p_m[1] * self.up_resolution + self.orig_h
        return (p_w_0, p_w_1)
