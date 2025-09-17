from shapely.geometry import Polygon
from scripts.base.my_utils import MyPoint, MyPath


class CellBase:
    """Base class for cells in the boustrophedon coverage path planning. """

    def __init__(self, ind: int, poly: Polygon):
        self.id: int = ind
        self.opt_id: int = None
        self.next_cell_seq: int = None
        #
        self.poly: Polygon = poly
        self.points: list[tuple[int, int]] = []
        #
        self.cxy: tuple[float, float] = None
        self.x_min: float = None
        self.x_max: float = None
        self.y_min: float = None
        self.y_max: float = None
        self.x_mid: float = None
        self.y_mid: float = None
        self.width: float = None
        self.height: float = None
        self.area: float = None
        #
        self.adjacent_ids: list[int] = []
        #
        self.do_planning: bool = True

        # start-end points
        self.start_point: MyPoint = None
        self.end_point: MyPoint = None

        # paths
        self.path_m: MyPath = None      # path in meters
        self.path_px: MyPath = None     # path in pixels

        # features
        self.rectangle_poly: Polygon = None       # inscribed rectangle polygon (expanded)
        self.rect_like: bool = False

        # calculate cell specs
        self.cell_specs()

        # create cell points
        self.create_cell_points()

    def set_optim_id(self, ind: int, next_cell_idx: int = None):
        self.opt_id = ind
        if next_cell_idx is not None:
            self.next_cell_seq = next_cell_idx
        else:
            self.next_cell_seq = None

    def cell_specs(self):
        # center
        self.cxy = self.poly.centroid.coords[0]
        # bounds
        minx, miny, maxx, maxy = self.poly.bounds
        self.x_min, self.y_min = minx, miny
        self.x_max, self.y_max = maxx, maxy
        # center
        self.x_mid = (self.x_min + self.x_max) / 2
        self.y_mid = (self.y_min + self.y_max) / 2
        # width and height
        self.width = maxx - minx
        self.height = maxy - miny
        # area
        self.area = self.poly.area

    def create_cell_points(self):
        # create cell points
        if self.poly:
            self.points = [(int(p[0]), int(p[1])) for p in self.poly.exterior.coords]
        else:
            self.points = []

    def add_adjacent_id(self, ind: int):
        self.adjacent_ids.append(ind)

    def set_path_m(self, path: MyPath):
        self.path_m = path

    def set_path_px(self, path: MyPath):
        self.path_px = path
        self.set_start_point(path.points[0])
        self.set_end_point(path.points[-1])

    def set_start_point(self, point: MyPoint):
        self.start_point = point

    def set_end_point(self, point: MyPoint):
        self.end_point = point
