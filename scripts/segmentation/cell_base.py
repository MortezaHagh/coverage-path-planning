"""Base cell class for segmentation and coverage path planning."""
import numpy as np
from shapely.geometry import Polygon
from scripts.base.my_utils import MyPoint, MyPath


class CellBase:
    """Base cell representation for decomposed workspace regions.

    This class provides the fundamental structure for cells created during
    space segmentation, storing geometric properties and adjacency information.
    """

    def __init__(self, idx: int, poly: Polygon):
        self.id: int = idx                  # cell id
        self.opt_id: int = None             # optimized id after sequencing
        self.next_cell_id_seq: int = None   # next cell id in the sequence

        # polygon and points
        self.poly: Polygon = poly                               # cell polygon
        self.points: np.ndarray = np.array([], dtype=np.int32)  # cell polygon points

        # cell specs
        self.cxy: tuple[float, float] = None  # center x,y
        self.x_min: float = None
        self.x_max: float = None
        self.y_min: float = None
        self.y_max: float = None
        self.x_mid: float = None
        self.y_mid: float = None
        self.width: float = None
        self.height: float = None
        self.area: float = None

        # cell specs
        self.major_direction: float = 0.0   # main direction angle of the cell (free space) - radians
        self.do_planning: bool = True       # whether to do planning in this cell

        # adjacency
        self.adjacent_ids: list[int] = []   # adjacent cell ids

        # cell paths (to be set later)
        self.path_m: MyPath = None      # path in meters
        self.path_px: MyPath = None     # path in pixels

        # start-end points (based on cell path)
        self.start_point: MyPoint = None
        self.end_point: MyPoint = None

        # calculate cell specs
        self.calculate_cell_specs()

        # create cell points
        self.create_cell_points()

    def set_optim_id(self, ind: int, next_cell_idx: int = None):
        """Set optimized sequence ID and next cell reference after sequencing."""
        self.opt_id = ind
        if next_cell_idx is not None:
            self.next_cell_id_seq = next_cell_idx
        else:
            self.next_cell_id_seq = None

    def calculate_cell_specs(self):
        """Calculate geometric properties: bounds, center, dimensions, and area."""
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
        """Extract integer coordinate points from cell polygon boundary."""
        if self.poly:
            points = [[int(p[0]), int(p[1])] for p in self.poly.exterior.coords]
            self.points = np.array(points, dtype=np.int32)
        else:
            self.points = []

    # --------------------------------------------------------------------------
    # setters
    # --------------------------------------------------------------------------

    def add_adjacent_id(self, ind: int):
        self.adjacent_ids.append(ind)

    def set_path_m(self, path: MyPath):
        """ set cell path in meters. """
        self.path_m = path

    def set_path_px(self, path: MyPath):
        """ set cell path in pixels. """
        self.path_px = path
        self.set_start_point(path.points[0])
        self.set_end_point(path.points[-1])

    def set_start_point(self, point: MyPoint):
        self.start_point = point

    def set_end_point(self, point: MyPoint):
        self.end_point = point
