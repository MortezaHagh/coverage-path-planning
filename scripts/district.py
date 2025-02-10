import numpy as np
from scripts.base.map_data import MapData
from scripts.gui.visuals import Rectangle


class ZoneMapData:
    map_data: MapData
    map_orig: np.ndarray
    map_cleaning: np.ndarray
    hw: tuple[int, int]
    bb_px: dict

    def __init__(self):
        self.map_orig = None
        self.map_cleaning = None
        self.hw = None
        self.bb_px = None

    def set_map_data(self, map_data: MapData):
        self.map_data = map_data

    def set_map_orig(self, map_orig: np.ndarray):
        self.map_orig = map_orig

    def set_map_cleaning(self, map_cleaning: np.ndarray):
        self.map_cleaning = map_cleaning
        self.hw = map_cleaning.shape

    def set_bb_px(self, bb_px: dict):
        self.bb_px = bb_px


class DistrictData:
    district_map_data: ZoneMapData

    def __init__(self, district_map_data: ZoneMapData):
        self.district_map_data = district_map_data

    # get zones rectangles
    # get zones map data
    # apply wf path on zones
    # apply cpp on zones

    def create_zone_rectangles(self):
        height, width = self.main_zone_map.hw
        x1 = width//2
        y1 = height//4
        x2 = width//2
        y2 = 3*height//4
        w1 = width
        h1 = height//2
        w2 = width
        h2 = height//2
        data1 = {"xc": x1, "yc": y1, "rot": 0, "w": w1, "h": h1}
        data2 = {"xc": x2, "yc": y2, "rot": 0, "w": w2, "h": h2}
        rect1 = Rectangle(data1)
        rect2 = Rectangle(data2)

        return rect1, rect2
