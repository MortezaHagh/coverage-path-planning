"""Map image processing module for binary map generation and morphological operations."""
import copy
import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.base.configs import MapConfig
from scripts.base.map_data import MapData


class MapImageProcessor:
    """Process grayscale map image and create structured MapData object.

    Args:
        map_arr: Loaded grayscale map image array (2D numpy array).
        map_config: Map configuration object with robot and workspace parameters.

    Attributes:
        map_data: Processed MapData object containing binary maps and metadata.

    Raises:
        ValueError: If image is not found, not grayscale, or format is unsupported.
    """

    def __init__(self, map_arr: np.ndarray, map_config: MapConfig):

        # logger
        self.lg = MyLogger("MapImageProcessor")
        self.lg.info1("Step: processing map image")

        # read configs and eval original map arr
        self.mcfg: MapConfig = map_config
        self.__orig_map: np.ndarray = map_arr
        self.check_map(map_arr)

        # set map data
        self.__map_data: MapData = MapData()
        self.set_map_data()

        # initial processes on map image - set binary map
        processed_map = self.process_map()
        self.__map_data.set_bin_map(processed_map)

        # add other data to map_data
        self.complete_map_data()

        # # visualize
        # self.visualize_map(self.processed_map)

    @property
    def map_data(self) -> MapData:
        return copy.deepcopy(self.__map_data)

    def check_map(self, image: np.ndarray) -> None:
        """Validate that the map image is properly loaded and formatted.

        Args:
            image: Input map image array.

        Raises:
            ValueError: If image is None or not grayscale.
        """
        if image is None:
            raise ValueError("Image not found or the format is not supported")
        if len(image.shape) > 2:
            raise ValueError("Image must be grayscale")

    def set_map_data(self) -> None:
        """Initialize basic map data attributes: name, origin, and original map."""
        map_arr = self.__orig_map.copy()
        self.__map_data.set_basic_configs(self.mcfg, self.mcfg.map_name)
        self.__map_data.set_origin_m(self.mcfg.origin_m)
        self.__map_data.set_orig_map(map_arr)

    def process_map(self) -> np.ndarray:
        """ create denoised binary map. """
        self.lg.info2("creating binary-denoised map")
        binary_map = self.convert_to_binary()
        denoised_map = self.remove_noise(binary_map)
        return denoised_map

    def convert_to_binary(self) -> np.ndarray:
        """ convert grayscale map to binary map. """
        # threshold = self.mcfg.occupied_thresh
        _, binary_map = cv.threshold(self.__orig_map, 220, 255, cv.THRESH_BINARY)
        return binary_map

    def remove_noise(self, binary_map: np.ndarray, kernel_size=(3, 3)) -> np.ndarray:
        """ remove noise from binary map using morphological opening. """
        kernel = np.ones(kernel_size, np.uint8)
        denoised_map = cv.morphologyEx(binary_map, cv.MORPH_OPEN, kernel)
        return denoised_map

    def complete_map_data(self) -> None:
        """ set resolution [and updated resolution] in map data. """
        self.__map_data.set_resolution(self.mcfg.resolution)

    def visualize_map(self, pgm_data_in: np.ndarray, disp_name: str = "Map") -> None:
        self.lg.info("display map image")

        # original map mage data
        map_data = pgm_data_in.copy()
        height, width = map_data.shape[:2]
        updated_h, updated_w = height, width
        self.lg.info("Map size [w,h]: {width}x{height}")

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
        self.lg.info("Image window closed.")
