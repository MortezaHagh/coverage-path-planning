import copy
import cv2 as cv
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import box, Point
from scripts.base.my_utils import mod_angle


class Rectangle:
    """ Rectangle class. 
        all data in pixels. 
    """
    rid: int = 0  # rectangle id

    def __init__(self, data: dict, rid: int = None):
        #
        self.ns: str = "Rectangle"
        self.point_radius: int = 40

        #
        self.x_min: int = 0
        self.y_min: int = 0

        # rid
        if rid is None:
            self.rid = Rectangle.rid
            Rectangle.rid += 1
        else:
            self.rid = rid

        # inputs
        xc = data["xc"]
        yc = data["yc"]
        rot = data["rot"]
        w = data["w"]
        h = data["h"]
        for v in data.values():
            if v is None:
                raise ValueError(f"[{self.ns}]:  input data is None")

        #
        self.__data: dict = data
        self.width: int = w
        self.height: int = h
        self.xc: float = xc
        self.yc: float = yc
        self.cp: tuple[float, float] = (xc, yc)
        self.rot: float = mod_angle(rot)
        # self.theta = np.atan2(w/h) + rot
        #
        self.xc_i: int = np.abs(xc).astype(int)
        self.yc_i: int = np.abs(yc).astype(int)
        self.cp_i: tuple[int, int] = (self.xc_i, self.yc_i)

        # create box
        bbox = box(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        bbox = rotate(bbox, np.rad2deg(self.rot), origin='center')
        self.box: box = bbox

        # vertices
        vertices_l = list(bbox.exterior.coords)[:4]
        vertices_l = np.array(vertices_l, dtype=np.int32)
        self.vertices_t: tuple[int, int] = tuple(vertices_l.tolist())
        self.vertices_a: np.ndarray = vertices_l.reshape((-1, 1, 2))
        self.vertices_ccw: np.ndarray = None

        # 4 points & rotation point
        self.tl: tuple[int, int] = None
        self.br: tuple[int, int] = None
        self.tr: tuple[int, int] = None
        self.bl: tuple[int, int] = None
        self.rot_point: tuple[int, int] = None
        self.up_all_points()
        self.cal_min_xy()

    @property
    def data(self) -> dict:
        return copy.copy(self.__data)

    @property
    def min_point(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min], dtype=np.int32)

    def update_data(self):
        #
        xc = self.xc
        yc = self.yc
        w = self.width
        h = self.height
        self.cp = (xc, yc)
        self.__data = {"xc": xc, "yc": yc, "rot": self.rot, "w": w, "h": h}
        # cp_i
        self.xc_i = np.abs(xc).astype(int)
        self.yc_i = np.abs(yc).astype(int)
        self.cp_i = (self.xc_i, self.yc_i)
        # self.__data = {"xc": self.xc_i, "yc": self.yc_i, "rot": self.rot, "w": w, "h": h}
        # create box
        bbox = box(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        rotated_bbox = rotate(bbox, np.rad2deg(self.rot), origin='center')
        self.box = rotated_bbox
        # vertices
        vertices_l = list(rotated_bbox.exterior.coords)
        vertices_l = np.array(vertices_l, dtype=np.int32)
        self.vertices_t = tuple(vertices_l.tolist())
        self.vertices_a = vertices_l.reshape((-1, 1, 2))
        # points
        self.up_all_points()

    def up_all_points(self):
        self.up_4_points()
        self.up_rotation_point()
        self.cal_min_xy()

    def up_rotation_point(self):
        xr = 0
        yr = - self.height / 2
        self.rot_point = self.rotate_point_t(xr, yr, self.rot)

    def up_4_points(self):
        tl = self.rotate_point_t(-self.width / 2, -self.height / 2, self.rot)
        br = self.rotate_point_t(self.width / 2, self.height / 2, self.rot)
        tr = self.rotate_point_t(self.width / 2, -self.height / 2, self.rot)
        bl = self.rotate_point_t(-self.width / 2, self.height / 2, self.rot)
        self.tl = tl
        self.br = br
        self.tr = tr
        self.bl = bl
        self.vertices_ccw = np.array([bl, br, tr, tl], dtype=np.int32)

    def rotate_point(self, dx, dy, rotation):
        x_r = dx * np.cos(rotation) - dy * np.sin(rotation)
        y_r = dx * np.sin(rotation) + dy * np.cos(rotation)
        return np.array([x_r + self.xc, y_r + self.yc], dtype=np.int32)

    def rotate_point_v2(self, x, y, rotation):
        x -= self.xc
        y -= self.yc
        x_r = x * np.cos(rotation) - y * np.sin(rotation)
        y_r = x * np.sin(rotation) + y * np.cos(rotation)
        return np.array([x_r + self.xc, y_r + self.yc], dtype=np.int32)

    def rotate_point_t(self, dx, dy, rotation):
        xy = self.rotate_point(dx, dy, rotation)
        return tuple(xy.tolist())

    def is_in_rect(self, x, y):
        point = Point(x, y)
        return self.box.contains(point)

    def are_close(self, p1, p2):
        return self.cal_distance_t(p1, p2) < self.point_radius

    def cal_distance_t(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def cal_min_xy(self):
        self.x_min = min(xy[0] for xy in self.vertices_t)
        self.y_min = min(xy[1] for xy in self.vertices_t)


class VisualRectangle(Rectangle):

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
