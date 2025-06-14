""" processing map pgm file """
import copy
import numpy as np
import cv2 as cv
from scripts.base.configs import BConfigs
from scripts.base.map_data import MapData


class MapImageProcessor:
    """ process map pgm file """
    ns: str
    __bcfg: BConfigs
    __orig_map: np.ndarray
    #
    __map_data = MapData

    def __init__(self, map_arr: np.ndarray, bconfigs: BConfigs):

        # init
        self.ns = "MapImageProcessor"
        print(f" >>>>> [{self.ns}] Step: processing map image")

        # read configs and eval original map arr
        self.__bcfg = bconfigs
        self.__orig_map = map_arr
        self.check_map(map_arr)

        # map data
        self.__map_data = MapData()
        self.set_map_data()

        # initial processes on map image - load data
        processed_map = self.process_map()
        self.__map_data.set_bin_map(processed_map)

        # add other data to map_data
        self.complete_data()

        # # visualize
        # self.visualize_map(self.processed_map)

    @property
    def mdata(self) -> MapData:
        return copy.deepcopy(self.__map_data)

    def check_map(self, image):
        """ check if map image is loaded """
        if image is None:
            raise ValueError("Image not found or the format is not supported")
        if len(image.shape) > 2:
            raise ValueError("Image must be grayscale")

    def set_map_data(self):
        print(f"[{self.ns}]: loading map image data")
        map_arr = self.__orig_map.copy()
        self.__map_data.set_basic_configs(self.__bcfg, self.__bcfg.map_name)
        self.__map_data.set_origin_m(self.__bcfg.origin_m)
        self.__map_data.set_orig_map(map_arr)

    def process_map(self) -> np.ndarray:
        print(f"[{self.ns}]: first processing of the map image")
        binary_map = self.convert_to_binary()
        denoised_map = self.remove_noise(binary_map)
        print(f"[{self.ns}]: Binary-denoised image created.")
        return denoised_map

    def convert_to_binary(self):
        print(f"[{self.ns}]: converting image to binary")
        # threshold = self.__bcfg.occupied_thresh
        _, binary_map = cv.threshold(self.__orig_map, 220, 255, cv.THRESH_BINARY)
        return binary_map

    def remove_noise(self, binary_map, kernel_size=(3, 3)) -> np.ndarray:
        print(f"[{self.ns}]: removing noise")
        kernel = np.ones(kernel_size, np.uint8)
        denoised_map = cv.morphologyEx(binary_map, cv.MORPH_OPEN, kernel)
        return denoised_map

    def complete_data(self):
        self.__map_data.set_resolution(self.__bcfg.resolution)

    def visualize_map(self, pgm_data_in: np.ndarray, disp_name: str = "Map") -> None:
        print(f"[{self.ns}]: display map image")

        # original map mage data
        map_data = pgm_data_in.copy()
        height, width = map_data.shape[:2]
        updated_h, updated_w = height, width
        print(f"[{self.ns}]: Map size [w,h]: {width}x{height}")

        # cv window
        wind_hw = self.__map_data.wind_hw
        zoom = min(wind_hw[0]/height, wind_hw[1]/width)
        updated_h = int(height * zoom)
        updated_w = int(width * zoom)
        updated_wh = (updated_w, updated_h)

        # display map image
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
