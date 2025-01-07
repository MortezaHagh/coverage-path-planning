""" processing map pgm file """
from typing import Optional
import copy
import numpy as np
import cv2 as cv
from scripts.map_data import MapData, Configs


class MapImageProcessor:
    """ process map pgm file """
    _omd = MapData
    _orig_map: np.ndarray

    def __init__(self, map_arr: np.ndarray, configs: Configs):

        # read configs and eval original map arr
        self._configs = configs
        self._orig_map = map_arr
        self.check_map(map_arr)

        # init
        self.ns = "IP"
        self.processed_map_1 = None

        # initial processes on map image - load data
        self.process_map()

        # map data
        self._omd = MapData()
        self.update_data()

        # # visualize
        # self.visualize_map(self.processed_map_1)

        # #
        # self.select_crop()

    @property
    def configs(self) -> Configs:
        return copy.deepcopy(self._configs)

    @property
    def mdata(self) -> MapData:
        return copy.deepcopy(self._omd)

    @property
    def orig_map(self) -> np.ndarray:
        return copy.deepcopy(self._orig_map)

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
        _, binary_map = cv.threshold(self._orig_map, threshold, 255, cv.THRESH_BINARY)
        return binary_map

    def remove_noise(self, binary_map, kernel_size=(3, 3)) -> np.ndarray:
        print(f"[{self.ns}]: removing noise")
        kernel = np.ones(kernel_size, np.uint8)
        denoised_map = cv.morphologyEx(binary_map, cv.MORPH_OPEN, kernel)
        return denoised_map

    def update_data(self):
        print(f"[{self.ns}]: loading map image data")
        map_arr = self.orig_map.copy()
        self._omd.set_orig_map(map_arr, self.configs.map_name)

    def visualize_map(self, pgm_data_in: Optional[np.ndarray], disp_name: str = "Map") -> None:
        print(f"[{self.ns}]: display map image")

        # original map mage data
        map_data = pgm_data_in.copy()
        height, width = map_data.shape[:2]
        updated_h, updated_w = height, width
        print(f"[{self.ns}]: Map size [w,h]: {width}x{height}")

        # cv window
        wind_hw = self._omd.wind_hw
        zoom = min(wind_hw[0]/height, wind_hw[1]/width)
        updated_h = int(height * zoom)
        updated_w = int(width * zoom)
        updated_wh = (updated_w, updated_h)
        # canvas = np.zeros((*wind_hw, 3), np.uint8)

        # display map image
        # # WINDOW_NORMAL - WINDOW_GUI_NORMAL - WINDOW_AUTOSIZE - WINDOW_FREERATIO - WINDOW_FULLSCREEN
        cv.namedWindow(disp_name, cv.WINDOW_NORMAL)
        resized_map = cv.resize(map_data, updated_wh, interpolation=cv.INTER_AREA)
        cv.imshow(disp_name, resized_map)

        while True:
            key = cv.waitKey(20)
            c2 = cv.getWindowProperty(disp_name, cv.WND_PROP_VISIBLE) < 1
            if key >= 0 or c2:
                break

        cv.destroyAllWindows()
        print(f"[{self.ns}]: Image window closed.")

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

        # eval vertices bound
        max_w = min(max_w, self._omd.updated_w)
        min_w = max(min_w, 0)
        max_h = min(max_h, self._omd.updated_h)
        min_h = max(min_h, 0)

        #
        cropped_map = copy.copy(self.processed_map_1[min_h:max_h, min_w:max_w])
        self.visualize_map(cropped_map, "cropped")
