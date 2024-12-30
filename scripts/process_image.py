from dataclasses import dataclass
from typing import Optional
import copy
import numpy as np
import cv2 as cv
from process_input import Configs


@dataclass
class MapData:
    """ map data """
    orig_map_arr: np.ndarray

    def __init__(self):
        self.name = None
        self.orig_height = None
        self.orig_width = None
        self.updated_height = None
        self.updated_width = None
        self.window_size = (1080, 1920)
        self.zoom = None
        #
        self.orig_map_arr = None


class MapImageProcessor:
    """ process map image """
    _md = MapData()
    orig_map_arr: np.ndarray

    @property
    def configs(self) -> Configs:
        return self._configs

    @property
    def mdata(self) -> MapData:
        return self._md

    def __init__(self, map_arr: np.ndarray, configs: Configs):

        # map data
        self.ns = "IP"
        self._md = MapData()

        # init
        self.orig_map_arr = None
        self.processed_map_1 = None

        # read configs and eval original map arr
        self._configs = configs
        self.orig_map_arr = map_arr
        self._md.orig_map_arr = map_arr
        self.check_map(map_arr)

        # initial processes on map imape
        self.process_map()
        self.update_data()

        # # visualize
        # self.visualize_map(self.processed_map_1)

        #
        self.select_crop()

    def check_map(self, image):
        """ check if map image is loaded """
        if image is None:
            raise ValueError("Image not found or the format is not supported")
        if len(image.shape) > 2:
            raise ValueError("Image must be grayscale")

    def process_map(self) -> None:
        print(f"[{self.ns}]: first processing of the map image")
        binary_map = self.convert_to_binary()
        denoised_map = self.remove_noise(binary_map)
        self.processed_map_1 = denoised_map
        print(f"[{self.ns}]: Binary-denoised image created.")

    def convert_to_binary(self):
        print(f"[{self.ns}]: converting image to binary")
        threshold = self.configs.occupied_thresh
        _, binary_map = cv.threshold(self.orig_map_arr, threshold, 255, cv.THRESH_BINARY)
        return binary_map

    def remove_noise(self, binary_map, kernel_size=(3, 3)) -> np.ndarray:
        print(f"[{self.ns}]: removing noise")
        kernel = np.ones(kernel_size, np.uint8)
        denoised_map = cv.morphologyEx(binary_map, cv.MORPH_OPEN, kernel)
        return denoised_map

    def update_data(self):
        # original map mage data
        map_data = self.processed_map_1.copy()
        height, width = map_data.shape[:2]
        self._md.orig_height = height
        self._md.orig_width = width
        self._md.updated_height, self._md.updated_width = height, width
        print(f"[{self.ns}]: Map size: {width}x{height}")

    def visualize_map(self, pgm_data_in: Optional[np.ndarray], disp_name: str = "Map") -> None:

        # original map mage data
        map_data = pgm_data_in.copy()
        height, width = map_data.shape[:2]
        updated_height, updated_width = height, width
        print(f"[{self.ns}]: Map size: {width}x{height}")

        # cv window
        window_size = self._md.window_size
        zoom = min(window_size[0]/height, window_size[1]/width)
        zoom = min(window_size[0]/height, window_size[1]/width)
        if zoom > 1.0:
            updated_height = int(height * zoom)
            updated_width = int(width * zoom)
        updated_size = (updated_width, updated_height)
        # canvas = np.zeros((*window_size, 3), np.uint8)

        # display map image
        # # WINDOW_NORMAL - WINDOW_GUI_NORMAL - WINDOW_AUTOSIZE - WINDOW_FREERATIO - WINDOW_FULLSCREEN
        cv.namedWindow(disp_name, cv.WINDOW_NORMAL)
        resized_map = cv.resize(map_data, updated_size, interpolation=cv.INTER_AREA)
        cv.imshow(disp_name, resized_map)

        while True:
            key = cv.waitKey(20)
            c2 = cv.getWindowProperty(disp_name, cv.WND_PROP_VISIBLE) < 1
            if key >= 0 or c2:
                break

        cv.destroyAllWindows()
        print("Image window closed.")

    def select_crop(self):
        corners = [
            [0, 0],
            [300, 0],
            [300, 200],
            [0, 200],
        ]
        corners = np.array(corners)
        max_w, max_h = np.max(corners, axis=0)
        min_w, min_h = np.min(corners, axis=0)
        print(max_w, max_h)
        print(min_w, min_h)

        # eval vertices bound
        max_w = min(max_w, self._md.updated_width)
        min_w = max(min_w, 0)
        max_h = min(max_h, self._md.updated_height)
        min_h = max(min_h, 0)

        #
        cropped_map = copy.copy(self.processed_map_1[min_h:max_h, min_w:max_w])
        self.visualize_map(cropped_map, "cropped")
