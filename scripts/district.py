import numpy as np
from scripts.map_data import MapData


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
    main_zone_map: ZoneMapData

    def __init__(self, main_zone_map: ZoneMapData):
        self.main_zone_map = main_zone_map
