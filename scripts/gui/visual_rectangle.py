""" Visual Rectangle class module for GUI interaction. """
import cv2 as cv
import numpy as np
from scripts.gui.rectangle import Rectangle


class VisualRectangle(Rectangle):
    """ Visual Rectangle class for GUI interaction.
        Inherits from Rectangle class and adds GUI interaction features."""

    def __init__(self, data: dict):
        super().__init__(data=data)
        self.ns = "VisualRectangle"

        #
        self.line_width = 10
        self.point_radius = 40
        self.default_color = (0, 0, 255)
        self.selected_color = (255, 255, 0)
        self.highlight_color = (255, 0, 0)
        self.highlight_color_fill = (255, 255, 255)

        #
        self.selected: bool = False
        self.highlight: bool = False

        #
        self.select_type: str = None

    def deselect(self):
        self.selected = False

    def is_in_box(self, x, y):
        is_inside = self.is_in_rect(x, y)
        if is_inside:
            self.highlight = True
            return True
        else:
            self.highlight = False
            return False

    def detect(self, x, y):
        detected = None
        if self.is_in_rect(x, y):
            if self.are_close(self.tl, (x, y)):
                detected = "top_left"
            elif self.are_close(self.tr, (x, y)):
                detected = "top_right"
            elif self.are_close(self.br, (x, y)):
                detected = "bottom_right"
            elif self.are_close(self.bl, (x, y)):
                detected = "bottom_left"
            elif self.are_close(self.cp, (x, y)):
                detected = "center"
            elif self.are_close(self.rot_point, (x, y)):
                detected = "rotation_point"
            else:
                detected = "rectangle"
        return detected

    def button_down(self, x, y):
        detected = self.detect(x, y)
        if detected is not None:
            self.select_type = detected
        if detected == "rectangle":
            self.selected = ~self.selected

    def button_up(self, x, y):
        self.select_type = None
        # self.calculate_boundaries()

    def button_move(self, x, y):
        select_type = self.select_type
        if select_type == "center":
            self.drag_center(x, y)
        elif select_type == "rotation_point":
            self.drag_rotation(x, y)
        elif select_type == "top_left":
            self.drag_top_left(x, y)
        elif select_type == "top_right":
            self.drag_top_right(x, y)
        elif select_type == "bottom_left":
            self.drag_bottom_left(x, y)
        elif select_type == "bottom_right":
            self.drag_bottom_right(x, y)

    def drag_center(self, x, y):
        self.xc = x
        self.yc = y
        self.update_data()

    def drag_rotation(self, x, y):
        sin_rot = (x - self.xc) / np.sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2)
        self.rot = np.arcsin(sin_rot)
        self.update_data()

    def drag_top_left(self, x, y):
        self.xc = int((x + self.br[0]) / 2)
        self.yc = int((y + self.br[1]) / 2)
        tl = self.rotate_point_v2(x, y, -self.rot)
        br = self.rotate_point_v2(self.br[0], self.br[1], -self.rot)
        self.width = br[0] - tl[0]
        self.height = br[1] - tl[1]
        self.update_data()

    def drag_top_right(self, x, y):
        self.xc = int((x + self.bl[0]) / 2)
        self.yc = int((y + self.bl[1]) / 2)
        tr = self.rotate_point_v2(x, y, -self.rot)
        bl = self.rotate_point_v2(self.bl[0], self.bl[1], -self.rot)
        self.width = tr[0] - bl[0]
        self.height = bl[1] - tr[1]
        self.update_data()

    def drag_bottom_left(self, x, y):
        self.xc = int((x + self.tr[0]) / 2)
        self.yc = int((y + self.tr[1]) / 2)
        bl = self.rotate_point_v2(x, y, -self.rot)
        tr = self.rotate_point_v2(self.tr[0], self.tr[1], -self.rot)
        self.width = tr[0] - bl[0]
        self.height = bl[1] - tr[1]
        self.update_data()

    def drag_bottom_right(self, x, y):
        self.xc = int((x + self.tl[0]) / 2)
        self.yc = int((y + self.tl[1]) / 2)
        br = self.rotate_point_v2(x, y, -self.rot)
        tl = self.rotate_point_v2(self.tl[0], self.tl[1], -self.rot)
        self.width = br[0] - tl[0]
        self.height = br[1] - tl[1]
        self.update_data()

    def draw(self, img: np.ndarray):
        color = self.default_color
        if self.highlight:
            # update color
            color = self.highlight_color
            # center
            cv.circle(img, self.cp_i, self.point_radius, self.highlight_color, -1)
            # rotation point
            cv.circle(img, self.rot_point, self.point_radius, self.highlight_color, -1)
            # 4 points
            for vertex in self.vertices_t:
                cv.circle(img, vertex, self.point_radius, self.highlight_color, -1)

        elif self.selected:
            color = self.selected_color

        # draw box
        cv.polylines(img, [self.vertices_a], True, color, self.line_width,)

    def get_rect(self) -> Rectangle:
        data = {"xc": self.xc, "yc": self.yc, "rot": self.rot, "w": self.width, "h": self.height}
        return Rectangle(data, self.rid)
