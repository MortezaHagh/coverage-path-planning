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
        self.center = (xc, yc)
        #
        self.xc_i = np.abs(xc).astype(int)
        self.yc_i = np.abs(yc).astype(int)
        self.rot = mod_angle(rot)

        # create box
        bbox = box(xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
        self.box = bbox

        # Get the coordinates of the vertices
        vertices = list(bbox.exterior.coords)
        vertices = np.array(vertices, dtype=np.int32)
        vertices2 = vertices.reshape((-1, 1, 2))

        #
        self.vertices = vertices
        self.vertices2 = vertices2


class VisualRectangle:
    rect: Rectangle

    def __init__(self, data: list):
        self.ns = "VisualRectangle"

        # configs
        self.line_width = 10
        self.point_radius = 40
        self.default_color = (0, 0, 255)
        self.highlight_color = (255, 0, 0)
        self.highlight_color_fill = (255, 255, 255)

        #
        self.highlight = False

        #
        self.rect = Rectangle(data=data)

    def is_in_box(self, x, y):
        point = Point(x, y)
        is_inside = self.rect.box.contains(point)
        if is_inside:
            self.highlight = True
        else:
            self.highlight = False

    def draw(self, img: np.ndarray):
        verts = self.rect.vertices2
        color = self.default_color
        if self.highlight:
            # update color
            color = self.highlight_color
            # draw points
            cv.circle(img, self.rect.center, self.point_radius, self.highlight_color, -1)
            for vertex in self.rect.vertices:
                cv.circle(img, tuple(vertex), self.point_radius, self.highlight_color, -1)
        # draw box
        cv.polylines(img, [verts], True, color, self.line_width,)
