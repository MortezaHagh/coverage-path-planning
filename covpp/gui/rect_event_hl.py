""" Rectangle Event Handler module."""
from pynput.keyboard import Key
import cv2 as cv
import numpy as np
from covpp.gui.visual_rectangle import VisualRectangle


class RecEventHandler:
    """ Rectangle Event Handler class.
        Handles mouse and keyboard events for rectangle manipulation.
    """

    def __init__(self, map_img: np.ndarray):
        self.ns: str = "RecEventHandler"
        self.wind_name: str = "Select"

        #
        self.__map_img: np.ndarray = map_img
        self.canvas: np.ndarray = np.zeros_like(map_img, dtype=np.uint8)

        # visualization objects
        self.main_obj: VisualRectangle = None
        self.objects: list[VisualRectangle] = []

        # selected and highlighted objects indices
        self.selected_ind: int = None
        self.highlighted_ind: int = None

        # keybord and mouse events
        # self.__ctrl_down: bool = False
        self.__button_down: bool = False

    def mouse_handle(self, event, x, y, flags, param):
        # mx, my = self.md.point_to_map(x, y)

        if event == cv.EVENT_MOUSEMOVE:
            self.on_mouse_move(x, y, flags)

        if event == cv.EVENT_LBUTTONDBLCLK:
            self.on_lb_db_click(x, y, flags)

        if event == cv.EVENT_LBUTTONDOWN:
            self.on_button_down(x, y, flags)

        if event == cv.EVENT_LBUTTONUP:
            self.on_button_up(x, y, flags)

    def on_mouse_move(self, x, y, flags):
        # if self.main_obj is None:
        #     return
        # self.main_obj.is_in_box(x, y)

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
        elif self.__button_down:
            if self.selected_ind < len(self.objects):
                self.objects[self.selected_ind].button_move(x, y)
            else:
                print(f"[{self.ns}]: selected_ind >= len(self.objects)")

        #
        # if self.navigation_click_position:
        #     self.move_center(x, y)

    def on_lb_db_click(self, x, y, flags):
        # print(f"[{self.ns}]: ldb: [x,y], [{(x, y)}]")
        params = {"xc": x, "yc": y, "rot": 0, "w": 200, "h": 200}
        main_obj = VisualRectangle(data=params)
        self.objects.append(main_obj)
        self.main_obj = main_obj
        #
        self.deselect()
        self.highlighted_ind = None

    def on_button_down(self, x, y, flags):
        # #
        # print("highlighted_ind", self.highlighted_ind)
        # print("selected_ind", self.selected_ind)

        #
        if self.highlighted_ind is not None:
            if self.highlighted_ind >= len(self.objects):
                print(f"[{self.ns}]: highlighted_ind >= len(self.objects)")
                self.highlighted_ind = None

        #
        if self.selected_ind is None:
            if self.highlighted_ind is not None:
                self.selected_ind = self.highlighted_ind
            else:
                self.selected_ind = None
        else:
            if self.highlighted_ind is not None:
                if self.highlighted_ind != self.selected_ind:
                    self.objects[self.selected_ind].deselect()
                    self.selected_ind = self.highlighted_ind

                #
        if self.selected_ind is not None:
            if self.selected_ind < len(self.objects):
                self.update_history()
                self.__button_down = True
                self.objects[self.selected_ind].button_down(x, y)
            else:
                print(f"[{self.ns}]: selected_ind >= len(self.objects)")
                self.selected_ind = None

    def on_button_up(self, x, y, flags):
        self.__button_down = False
        if self.selected_ind is not None:
            try:
                self.objects[self.selected_ind].button_up(x, y)
                self.selected_ind = None
            except IndexError:
                if self.selected_ind >= len(self.objects):
                    print(f"[{self.ns}]: selected_ind >= len(self.objects)")
                raise
        else:
            self.selected_ind = None

    def on_key_press(self, key):
        if key == Key.ctrl:
            self.__ctrl_down = True
        elif key == Key.esc or key == Key.enter:
            return False

        if key == Key.ctrl:
            print(f"[{self.ns}]: crtls key is down")

    def on_key_release(self, key):
        if key == Key.esc or key == Key.enter:
            return False

        if key == Key.ctrl:
            self.__ctrl_down = False

    def update_history(self):
        pass

    def deselect(self):
        if self.selected_ind is not None:
            self.objects[self.selected_ind].deselect()
            self.selected_ind = None

    def update_image(self):
        self.canvas[:, :] = self.__map_img
        if self.main_obj is None:
            return
        for ob in self.objects:
            ob.draw(self.canvas)
        # self.main_obj.draw(self.canvas)

    def get_vis_objects(self, selected=False):
        if selected:
            vos = [obj for obj in self.objects if obj.selected]
        else:
            vos = self.objects
        #
        if not vos:
            print(f"[{self.ns}]: There is no object to return!")
        return vos

    def get_main_obj(self) -> VisualRectangle:
        if self.main_obj is None:
            raise ValueError(f"[{self.ns}]: main object is has not created!")
        return self.main_obj
