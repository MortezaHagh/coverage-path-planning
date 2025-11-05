""" Utility classes and functions. """
import dataclasses
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


@dataclasses.dataclass
class MyPoint:
    """ simple point class x, y. Represents a point in 2D space. """

    x: float = None
    y: float = None

    @property
    def xy(self):
        return (self.x, self.y)


class MyPath:
    """ simple path class - list of points (MyPoint). """

    def __init__(self, points: list[MyPoint] = None):
        self.points: list[MyPoint] = points
        self.xy: np.ndarray = []
        self.n: int = 0
        self.set_xys()

    def set_points(self, points: list[MyPoint]) -> None:
        self.points = points
        self.set_xys()

    def set_xys(self):
        """ create xy array (list of tuples) from points. """
        if self.points is not None:
            xy = [(p.x, p.y) for p in self.points]
            if isinstance(xy[0][0], (float, np.float32)):
                self.xy = np.array(xy, dtype=np.float32)
            else:
                self.xy = np.array(xy, dtype=np.int32)
            self.n = len(self.points)

    def point(self, i: int) -> MyPoint:
        """ get point at index i. """
        if i < 0 or i >= self.n:
            raise IndexError("Index out of range")
        return self.points[i]


# ----------------------------------------------------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------------------------------------------------


def get_n_colors_plt(n: int) -> list:
    """ Get n distinct colors from matplotlib's colormaps or HSV spectrum. """
    cmap = plt.get_cmap('hsv')  # You can use 'tab10', 'tab20', 'Set3', etc.
    colors_rgb = [cmap(i / n)[:3] for i in range(n)]  # RGB values (0-1)
    return colors_rgb


def get_n_colors_cv(n: int) -> list:
    """ Get n distinct colors in BGR format for OpenCV. """
    colors_rgb = get_n_colors_plt(n)
    # Convert to BGR and scale to 0-255
    colors_bgr = [tuple(int(255 * c) for c in reversed(rgb)) for rgb in colors_rgb]
    return colors_bgr


def mod_angle(theta: float) -> float:
    return theta % (2 * np.pi)


def visualize_map(pgm_data_in: np.ndarray, name=None) -> None:
    # ns = "utils"
    # print(f"[{ns}]: display map image")

    # check if the image is loaded
    if pgm_data_in is None:
        print("Failed to load PGM file.")
        return

    # original image data
    map_data = pgm_data_in.copy()
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


def visualize_map_s(pgm_data_in: np.ndarray, w_name=None, wait=True) -> None:
    if w_name is None:
        w_name = "PGM Image"
    cv.namedWindow(w_name, cv.WINDOW_NORMAL)
    cv.imshow(w_name, pgm_data_in)
    if wait:
        cv.waitKey()
        cv.destroyAllWindows()
