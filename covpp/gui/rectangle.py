""" Rectangle class module. """
import copy
import numpy as np
from shapely.affinity import rotate
from shapely.geometry import box, Point
from covpp.base.geometry import mod_angle


class Rectangle:
    """ Rectangle class.
        all data in pixels.
        Includes methods to manipulate rectangle points and rotation.
    """

    rid: int = 0  # rectangle id

    def __init__(self, data: dict, rid: int = None):
        self.ns: str = "Rectangle"
        self.point_radius: int = 40

        # min x,y
        self.x_min: int = 0
        self.y_min: int = 0
        self.x_max: int = 0
        self.y_max: int = 0

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

        # initialize attributes
        self.__data: dict = data
        self.width: int = w  # width of the rectangle
        self.height: int = h
        self.xc: float = xc
        self.yc: float = yc
        self.cp: tuple[float, float] = (xc, yc)
        self.rot: float = mod_angle(rot)
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

    def update_data(self):
        """ Update rectangle data dictionary. """
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

    def up_rotation_point(self):
        xr = 0
        yr = - self.height / 2
        self.rot_point = self.rotate_point_t(xr, yr, self.rot)

    def cal_min_xy(self):
        self.x_min = min(xy[0] for xy in self.vertices_t)
        self.y_min = min(xy[1] for xy in self.vertices_t)
        self.x_max = max(xy[0] for xy in self.vertices_t)
        self.y_max = max(xy[1] for xy in self.vertices_t)

    # --------------------------------------------------------------------------
    # Methods for point rotation and distance calculations
    # --------------------------------------------------------------------------

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
