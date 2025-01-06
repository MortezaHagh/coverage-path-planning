from pynput.keyboard import Key
import cv2 as cv
import numpy as np
from scripts.map_data import Configs
from scripts.gui.visuals import VisualRectangle


class RecEventHandler:
    img: np.ndarray
    vis_obj: VisualRectangle
    objects: list[VisualRectangle]
    #
    selected_ind: int
    highlighted_ind: int

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
        self.vis_obj: VisualRectangle = None
        self.objects: list[VisualRectangle] = []

        #
        self.selected_ind: int = None
        self.highlighted_ind: int = None

        #
        self.ctrl_down = False
        self.button_down = False

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
        # if self.vis_obj is None:
        #     return
        # self.vis_obj.is_in_box(x, y)

        #
        highlighted: bool = False
        if self.selected_ind is None:
            if self.highlighted_ind is None:
                for i, ob in enumerate(self.objects):
                    if highlighted:
                        ob.highlight = False
                        continue
                    if ob.is_in_box(x, y):
                        self.highlighted_ind = i
                        highlighted = True
            else:
                if self.highlighted_ind < len(self.objects):
                    if not self.objects[self.highlighted_ind].is_in_box(x, y):
                        self.highlighted_ind = None
                else:
                    self.highlighted_ind = None
                    print(f"[{self.ns}]: highlighted_ind >= len(self.objects)")
        elif self.button_down:
            if self.selected_ind < len(self.objects):
                self.objects[self.selected_ind].button_move(x, y)
            else:
                print(f"[{self.ns}]: selected_ind >= len(self.objects)")

        #
        # if self.navigation_click_position:
        #     self.move_center(x, y)

    def on_lb_db_click(self, x, y, flags):
        print(f"[{self.ns}]: lb_db clicked")
        print(f"[{self.ns}]: [x,y], [{(x, y)}]")
        params = [x, y, 200, 200, 0]
        vis_obj = VisualRectangle(data=params)
        self.vis_obj = vis_obj
        self.objects.append(vis_obj)
        #
        self.selected_ind = None
        self.highlighted_ind = None

    def on_button_down(self, x, y, flags):
        # if self.vis_obj is None:
        #     return
        # if self.vis_obj.highlight:
        #     print("ooooooooooo")

        if self.selected_ind is None:
            if self.highlighted_ind is not None:
                if self.highlighted_ind < len(self.objects):
                    self.selected_ind = self.highlighted_ind
                else:
                    print(f"[{self.ns}]: highlighted_ind >= len(self.objects)")
                    self.selected_ind = None
            else:
                self.selected_ind = None
        #
        if self.selected_ind is not None:
            if self.selected_ind < len(self.objects):
                self.update_history()
                self.button_down = True
                self.objects[self.selected_ind].button_down(x, y)
            else:
                print(f"[{self.ns}]: selected_ind >= len(self.objects)")
                self.selected_ind = None

    def on_button_up(self, x, y, flags):
        if self.selected_ind is not None:
            try:
                self.objects[self.selected_ind].button_up(x, y)
                self.selected_ind = None
            except:
                if self.selected_ind >= len(self.objects):
                    print(f"[{self.ns}]: selected_ind >= len(self.objects)")
                raise
        else:
            self.selected_ind = None

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

    def update_history(self):
        pass

    def update_image(self):
        self.canvas[:, :] = self.map_img
        if self.vis_obj is None:
            return
        for ob in self.objects:
            ob.draw(self.canvas)
        # self.vis_obj.draw(self.canvas)
