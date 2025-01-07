from typing import Optional
import numpy as np
import cv2 as cv


def mod_angle(theta):
    return theta % (2 * np.pi)


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
