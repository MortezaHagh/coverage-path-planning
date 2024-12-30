from typing import Optional
import numpy as np
import cv2 as cv
from process_input import Configs


class ImageProcessor:
    orig_map_arr: np.ndarray

    def __init__(self, map_arr: np.ndarray, configs: Configs):

        # map data
        self.orig_height = None
        self.orig_width = None
        self.window_size = None
        self.zoom = None

        #
        self.configs = configs
        self.orig_map_arr = map_arr
        self.check_image(map_arr)

        # #
        # binary_image = self.convert_to_binary(threshold=self.configs.occupied_thresh)
        # print("Binary image created.")
        #
        self.visualize_map(map_arr)

    def check_image(self, image):
        if image is None:
            raise ValueError("Image not found or the format is not supported")
        if len(image.shape) > 2:
            raise ValueError("Image must be grayscale")

    def convert_to_binary(self, threshold=128):
        _, binary_image = cv.threshold(self.orig_map_arr, threshold, 255, cv.THRESH_BINARY)
        return binary_image

    def remove_noise(self, binary_image, kernel_size=(3, 3)):
        kernel = np.ones(kernel_size, np.uint8)
        denoised_image = cv.morphologyEx(binary_image, cv.MORPH_OPEN, kernel)
        return denoised_image

    def process_image(self):
        binary_image = self.convert_to_binary()
        denoised_image = self.remove_noise(binary_image)
        return denoised_image

    def visualize_map(self, pgm_data_in: Optional[np.ndarray]) -> None:
        # check if the image is loaded
        if pgm_data_in is None:
            print("Failed to load PGM file.")
            return

        # original image data
        map_data = pgm_data_in.copy()
        height, width = map_data.shape[:2]
        updated_height, updated_width = height, width
        self.orig_height = height
        self.orig_width = width
        print(f"Map size: {width}x{height}")

        # cv window
        w_name = "PGM Image"
        window_size = (1080, 1920)
        zoom = min(window_size[0]/height, window_size[1]/width)
        self.window_size = window_size
        self.zoom = zoom
        zoom = min(window_size[0]/height, window_size[1]/width)
        if zoom > 1.0:
            updated_height = int(height * zoom)
            updated_width = int(width * zoom)
        updated_size = (updated_width, updated_height)
        # canvas = np.zeros((*window_size, 3), np.uint8)

        # display image
        # # WINDOW_NORMAL - WINDOW_GUI_NORMAL - WINDOW_AUTOSIZE - WINDOW_FREERATIO - WINDOW_FULLSCREEN
        cv.namedWindow(w_name, cv.WINDOW_NORMAL)
        resized_image = cv.resize(map_data, updated_size, interpolation=cv.INTER_AREA)
        cv.imshow(w_name, resized_image)

        while True:
            key = cv.waitKey(20)
            c2 = cv.getWindowProperty(w_name, cv.WND_PROP_VISIBLE) < 1
            if key >= 0 or c2:
                break

        cv.destroyAllWindows()
        print("Image window closed.")
