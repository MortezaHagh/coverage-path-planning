import cv2 as cv
import numpy as np
from shapely.geometry import box, Point
from scripts.my_utils import mod_angle


class Rectangle:
    bbox: box

    def __init__(self, data: list):
        #
        self.ns = "Rectangle"
        xc = data[0]
        yc = data[1]
        rot = data[4]
        w = data[2]
        h = data[3]

        #
        self.width = w
        self.height = h
        self.xc = xc
        self.yc = yc
        self.cp = (xc, yc)
        #
        self.xc_i = np.abs(xc).astype(int)
        self.yc_i = np.abs(yc).astype(int)
        self.cp_i = (self.xc_i, self.yc_i)
        self.rot = mod_angle(rot)

        # create box
        bbox = box(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        self.box = bbox

        # vertices
        vertices_l = list(bbox.exterior.coords)
        vertices_a = np.array(vertices_l, dtype=np.int32)
        self.vertices_t = tuple(vertices_a.tolist())
        self.vertices_a = vertices_a.reshape((-1, 1, 2))

        # 4 points & rotation point
        self.tl: tuple[int, int] = None
        self.br: tuple[int, int] = None
        self.tr: tuple[int, int] = None
        self.bl: tuple[int, int] = None
        self.rot_point: tuple[int, int] = None
        self.up_all_points()

    def update_data(self, xc, yc, w, h, rot=0.0):
        self.width = w
        self.height = h
        # cp
        self.xc = xc
        self.yc = yc
        self.cp = (xc, yc)
        # cp_i
        self.xc_i = np.abs(xc).astype(int)
        self.yc_i = np.abs(yc).astype(int)
        self.rot = mod_angle(rot)
        self.cp_i = (self.xc_i, self.yc_i)
        # create box
        bbox = box(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        self.box = bbox
        # vertices
        vertices_l = list(bbox.exterior.coords)
        vertices_a = np.array(vertices_l, dtype=np.int32)
        self.vertices_t = tuple(vertices_a.tolist())
        self.vertices_a = vertices_a.reshape((-1, 1, 2))
        # points
        self.up_all_points()

    def up_all_points(self):
        self.up_4_points()
        self.up_rotation_point()

    def up_rotation_point(self):
        xr = 0
        yr = - self.height / 2
        self.rot_point = self.rotate_point_i(xr, yr, self.rot)

    def up_4_points(self):
        tl = self.rotate_point_i(-self.width / 2, -self.height / 2, self.rot)
        br = self.rotate_point_i(self.width / 2, self.height / 2, self.rot)
        tr = self.rotate_point_i(self.width / 2, -self.height / 2, self.rot)
        bl = self.rotate_point_i(-self.width / 2, self.height / 2, self.rot)
        self.tl = tl
        self.br = br
        self.tr = tr
        self.bl = bl
        return [tl, br, tr, bl]

    def rotate_point(self, dx, dy, rotation):
        x_r = dx * np.cos(rotation) - dy * np.sin(rotation)
        y_r = dx * np.sin(rotation) + dy * np.cos(rotation)
        return np.array([x_r + self.xc, y_r + self.yc], dtype=np.int32)

    def rotate_point_i(self, dx, dy, rotation):
        xy = self.rotate_point(dx, dy, rotation)
        return tuple(xy.tolist())


class VisualRectangle:
    rect: Rectangle

    def __init__(self, data: list):
        self.ns = "VisualRectangle"

        # configs
        self.line_width = 10
        self.point_radius = 40
        self.default_color = (0, 0, 255)
        self.selected_color = (255, 255, 0)
        self.highlight_color = (255, 0, 0)
        self.highlight_color_fill = (255, 255, 255)

        #
        self.selected = False
        self.highlight = False

        #
        self.point_highlight_type = None

        #
        self.rect = Rectangle(data=data)

    def is_in_box(self, x, y):
        is_inside = self.is_in_rect(x, y)
        if is_inside:
            self.highlight = True
            return True
        else:
            self.highlight = False
            return False

    def is_in_rect(self, x, y):
        point = Point(x, y)
        return self.rect.box.contains(point)

    def button_down(self, x, y):
        detected = self.detect(x, y)
        if detected is not None:
            self.point_highlight_type = detected
        if detected == "rectangle":
            self.selected = ~self.selected

    def button_up(self, x, y):
        self.point_highlight_type = None
        # self.calculate_boundaries()

    def button_move(self, x, y):
        print("move")

    def detect(self, x, y):
        detected = None
        if self.is_in_rect(x, y):
            if self.are_close(self.rect.tl, (x, y)):
                detected = "top_left"
            elif self.are_close(self.rect.tr, (x, y)):
                detected = "top_right"
            elif self.are_close(self.rect.br, (x, y)):
                detected = "bottom_right"
            elif self.are_close(self.rect.bl, (x, y)):
                detected = "bottom_left"
            elif self.are_close(self.rect.cp, (x, y)):
                detected = "center"
            elif self.are_close(self.rect.rot_point, (x, y)):
                detected = "rotation_point"
            else:
                detected = "rectangle"
        return detected

    def are_close(self, p1, p2):
        return self.cal_distance(p1, p2) < self.point_radius

    def cal_distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def draw(self, img: np.ndarray):
        color = self.default_color
        if self.highlight:
            # update color
            color = self.highlight_color
            # center
            cv.circle(img, self.rect.cp, self.point_radius, self.highlight_color, -1)
            # rotation point
            cv.circle(img, self.rect.rot_point, self.point_radius, self.highlight_color, -1)
            # 4 points
            for vertex in self.rect.vertices_t:
                cv.circle(img, vertex, self.point_radius, self.highlight_color, -1)

        elif self.selected:
            color = self.selected_color

        # draw box
        cv.polylines(img, [self.rect.vertices_a], True, color, self.line_width,)
