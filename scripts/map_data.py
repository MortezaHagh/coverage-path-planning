from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2 as cv


@dataclass
class Configs:
    """ All configs """

    def __init__(self):
        # robor configs
        self.robot_radius = 0.5
        self.safe_distance = 0.2
        # directions
        self.map_name = None
        self.map_path = None
        self.data_dir = None
        # map configs
        self.origin = None
        self.resolution = None
        self.free_thresh = None
        self.occupied_thresh = None


class MapData:
    """ map data """
    orig_map: np.ndarray
    up_map: np.ndarray
    up_rgb_map: np.ndarray

    def __init__(self):
        self.ns = "MapData"
        #
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
        #
        self.wind_hw = (1080, 1920)
        self.wind_h = None
        self.wind_w = None
        self.wind_center = None
        self.zoom = None
        #
        self.orig_map = None
        self.up_map = None
        self.up_rgb_map = None

    def set_orig_map(self, map_data: np.ndarray, name: str = None):
        #
        self.name = name
        self.orig_map = map_data

        #
        self.set_map_hw()
        self.set_wind_wh()
        self.resize_map()
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
        self.set_updated_wh(h=updated_h, w=updated_w)

    def set_updated_wh(self, w, h):
        self.updated_h = h
        self.updated_w = w
        self.updated_wh = (w, h)
        print(f"[{self.ns}]: Updated Map size [w,h]: {w}x{h}")

    def resize_map(self):
        self.up_map = cv.resize(self.orig_map, self.updated_wh, interpolation=cv.INTER_AREA)
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


def visualize_map(pgm_data_in: Optional[np.ndarray]) -> None:
    ns = "utils"
    print(f"[{ns}]: display map image")

    # check if the image is loaded
    if pgm_data_in is None:
        print("Failed to load PGM file.")
        return

    # original image data
    map_data = pgm_data_in.copy()
    height, width = map_data.shape[:2]
    updated_h, updated_w = height, width
    print(f"[{ns}]: Map size: {width}x{height}")

    # cv window
    w_name = "PGM Image"
    wind_hw = (1080, 1920)
    zoom = min(wind_hw[0]/height, wind_hw[1]/width)
    if zoom > 1.0:
        updated_h = int(height * zoom)
        updated_w = int(width * zoom)
    updated_wh = (updated_w, updated_h)

    # display image
    # # WINDOW_NORMAL - WINDOW_GUI_NORMAL - WINDOW_AUTOSIZE - WINDOW_FREERATIO - WINDOW_FULLSCREEN
    cv.namedWindow(w_name, cv.WINDOW_NORMAL)
    resized_image = cv.resize(map_data, updated_wh, interpolation=cv.INTER_AREA)
    cv.imshow(w_name, resized_image)

    while True:
        key = cv.waitKey(20)
        c2 = cv.getWindowProperty(w_name, cv.WND_PROP_VISIBLE) < 1
        if key >= 0 or c2:
            break

    cv.destroyAllWindows()
    print(f"[{ns}]: Image window closed.")
