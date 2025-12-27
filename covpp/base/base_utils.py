""" Utility functions. """
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def get_n_colors_plt(n: int) -> list:
    """ Get n distinct colors from matplotlib's colormaps or HSV spectrum. """
    if n <= 0:
        raise ValueError("Number of colors 'n' must be a positive integer.")
    cmap = plt.get_cmap('Dark2')  # You can use 'tab10', 'tab20', 'Set3', 'Dark2'.
    colors_rgb = [cmap(i / n)[:3] for i in range(n)]  # RGB values (0-1)
    return colors_rgb


def get_n_colors_cv(n: int) -> list:
    """ Get n distinct colors in BGR format for OpenCV. """
    if n <= 0:
        raise ValueError("Number of colors 'n' must be a positive integer.")
    colors_rgb = get_n_colors_plt(n)
    # Convert to BGR and scale to 0-255
    colors_bgr = [tuple(int(255 * c) for c in reversed(rgb)) for rgb in colors_rgb]
    return colors_bgr


def visualize_map(map_image: np.ndarray, name=None) -> None:
    """ Visualize PGM map data using OpenCV. """

    # check map_image
    if map_image is None:
        raise ValueError("Input map_image is None.")

    # original image data
    map_data = map_image.copy()
    height, width = map_data.shape[:2]
    updated_h, updated_w = height, width
    # print(f"[{ns}]: Map size: {width}x{height}")

    # cv window
    w_name = name
    if name is None:
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
    # print(f"[{ns}]: Image window closed.")


def visualize_map_s(map_image: np.ndarray, w_name=None, wait=True) -> None:
    """ Visualize PGM map data using OpenCV (simple version). """
    if map_image is None:
        raise ValueError("Input map_image is None.")
    if w_name is None:
        w_name = "PGM Image"
    cv.namedWindow(w_name, cv.WINDOW_NORMAL)
    cv.imshow(w_name, map_image)
    if wait:
        cv.waitKey()
        cv.destroyAllWindows()
