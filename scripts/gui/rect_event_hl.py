from pynput.keyboard import Key
import cv2 as cv
import numpy as np
from scripts.map_data import Configs
from scripts.gui.visuals import VisualRectangle


class RecEventHandler:
    vis_obj: VisualRectangle
    img: np.ndarray

    def __init__(self, map_img: np.ndarray, configs: Configs):
        #
        self.ns = "RecEventHandler"
        self.wind_name = "Select"

        #
        self.cf = configs
        self.map_img = map_img
        self.canvas = np.zeros_like(map_img, dtype=np.uint8)
        # self.md = map_data
        # self.img = map_data.up_rgb_map

        #
        self.selecteded_ind = None
        self.highlighted_ind = None

        #
        self.ctrl_down = False

        #
        self.vis_obj = None  # []

    def mouse_handle(self, event, x, y, flags, param):
        # mx, my = self.md.point_to_map(x, y)

        #
        if event == cv.EVENT_MOUSEMOVE:
            self.on_mouse_move(x, y, flags)

        if event == cv.EVENT_LBUTTONDBLCLK:
            self.on_lb_db_click(x, y, flags)

        if event == cv.EVENT_LBUTTONDOWN:
            self.on_button_down(x, y, flags)

        if event == cv.EVENT_LBUTTONUP:
            self.on_button_up(x, y, flags)

    def on_mouse_move(self, x, y, flags):
        if self.vis_obj is None:
            return
        self.vis_obj.is_in_box(x, y)

    def on_lb_db_click(self, x, y, flags):
        print(f"[{self.ns}]: lb_db clicked")
        print(f"[{self.ns}]: [x,y], [{(x, y)}]")
        params = [x, y, 200, 200, 0]
        self.vis_obj = VisualRectangle(data=params)

    def on_button_down(self, x, y, flags):
        if self.vis_obj is None:
            return
        if self.vis_obj.highlight:
            print("ooooooooooo")

    def on_button_up(self, x, y, flags):
        # if self.chosen_object_idx is not None and self.chosen_object_idx < len(self.objects):
        #     self.objects[self.chosen_object_idx].button_up(x, y)
        #     self.chosen_object_idx = None
        # else:
        #     self.chosen_object_idx = None
        pass

    def on_press(self, key):
        print(f"[{self.ns}]: a key is down")

        if key == Key.ctrl:
            self.ctrl_down = True
        elif key == Key.esc or key == Key.enter:
            return False

        if key == Key.ctrl:
            print(f"[{self.ns}]: crtls key is down")

    def on_release(self, key):
        print(f"[{self.ns}]: key is released")
        if key == Key.esc or key == Key.enter:
            return False

        if key == Key.ctrl:
            self.ctrl_down = False

    def update_image(self):
        self.canvas[:, :] = self.map_img
        if self.vis_obj is None:
            return
        self.vis_obj.draw(self.canvas)
