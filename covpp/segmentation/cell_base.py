"""Base cell class for segmentation and coverage path planning."""
from copy import deepcopy
import numpy as np
from shapely.geometry import Polygon
from shapely import maximum_inscribed_circle  # , concave_hull
from covpp.base.logger import MyLogger
from covpp.base.configs import PlanningConfig
from covpp.base.geometry import MyPoint, MyPath


class CellBase:
    """Base cell representation for decomposed workspace regions.

    This class provides the fundamental structure for cells created during
    space segmentation, storing geometric properties and adjacency information.

    Args:
        idx (int): Unique identifier for the cell.
        poly (Polygon): Shapely polygon defining the cell boundary.
        resolution (float): Map resolution in meters per pixel.
        simplify (bool): Whether to simplify the polygon upon creation.

    Attributes:
        id (int): Cell identifier.
        opt_id (int): Optimized ID after sequencing.
        next_cell_id (int): Next cell ID in the sequence.
        resolution (float): Map resolution.
        base_poly (Polygon): Original cell polygon before modifications.
        poly (Polygon): Current cell polygon.
        points (np.ndarray): Array of integer coordinate points of the cell boundary.
        cxy (tuple[float, float]): Center coordinates of the cell.
        x_min_p, x_max_p, y_min_p, y_max_p (float): Cell boundary extents.
        x_mid_p, y_mid_p (float): Midpoint coordinates of the cell.
        width (float): Width of the cell.
        height (float): Height of the cell.
        area (float): Area of the cell.
        size_type (str): Size classification ("small", "normal", "large").
        slim_type (str): Slimness classification ("slim", "normal").
        major_direction (float): Main direction angle of the cell in radians.
        do_planning (bool): Flag indicating if planning should be performed in this cell.
        adjacent_ids (list[int]): List of adjacent cell IDs.
        path_m (MyPath): Coverage path in meters.
        path_px (MyPath): Coverage path in pixels.
        start_point (MyPoint): Starting point of the cell path.
        end_point (MyPoint): Ending point of the cell path.

    """

    def __init__(self, idx: int, poly: Polygon, resolution: float, simplify: bool = True):
        self.lg = MyLogger(f"CellBase-{idx}")
        self.id: int = idx

        # optimization attributes
        self.opt_id: int = None                 # optimized id after sequencing
        self.next_cell_id: int = None       # next cell id in the sequence

        # settings
        self.resolution: float = resolution
        self.robot_r_px = int(PlanningConfig.cleaning_r_ov / resolution)
        self.small_cell_thresh_c = 12.0         # small cell threshold coefficient
        self.slim_cell_thresh_c = 2.0           # slim cell threshold coefficient

        # polygon and points
        if simplify:
            poly = self.simplify_cell_poly(poly)
        self.base_poly: Polygon = deepcopy(poly)                # original cell polygon before any modifications
        self.poly: Polygon = poly                               # cell polygon
        self.points: np.ndarray = np.array([], dtype=np.int32)  # cell polygon points

        # cell specs
        self.cxy: tuple[float, float] = None  # center x,y
        self.x_min_p: float = None
        self.x_max_p: float = None
        self.y_min_p: float = None
        self.y_max_p: float = None
        self.x_mid_p: float = None
        self.y_mid_p: float = None
        self.width: float = None
        self.height: float = None
        self.area: float = None

        # cell classification
        self.size_type: str = "normal"      # small, normal, large
        self.slim_type: str = "normal"      # slim, normal

        # cell specs
        self.major_direction: float = 0.0   # main direction angle of the cell (free space) - radians
        self.do_planning: bool = True       # whether to do planning in this cell

        # adjacency
        self.adjacent_ids: list[int] = []   # adjacent cell ids

        # cell paths (to be set later)
        self.path_m: MyPath = None          # path in meters
        self.path_px: MyPath = None         # path in pixels

        # start-end points (based on cell path)
        self.start_point: MyPoint = None
        self.end_point: MyPoint = None

        # ----------------------------------------------------------------------
        # calculate cell specs
        self.calculate_cell_specs()
        self.evaluate_cell_area()
        self.eval_cell_slimness()

        # create cell points
        self.create_cell_points()

    def set_optim_id(self, ind: int, next_cell_idx: int = None):
        """Set optimized sequence ID and next cell reference after sequencing."""
        self.opt_id = ind
        if next_cell_idx is not None:
            self.next_cell_id = next_cell_idx
        else:
            self.next_cell_id = None

    def calculate_cell_specs(self):
        """Calculate geometric properties: bounds, center, dimensions, and area."""
        # center
        self.cxy = self.poly.centroid.coords[0]
        # bounds
        minx, miny, maxx, maxy = self.poly.bounds
        self.x_min_p, self.y_min_p = minx, miny
        self.x_max_p, self.y_max_p = maxx, maxy
        # center
        self.x_mid_p = (self.x_min_p + self.x_max_p) / 2
        self.y_mid_p = (self.y_min_p + self.y_max_p) / 2
        # width and height
        self.width = maxx - minx
        self.height = maxy - miny
        # area
        self.area = self.poly.area

    def evaluate_cell_area(self) -> None:
        """ Evaluate cell area and classify size. """
        # classify small cells
        robot_area_px = np.pi * (self.robot_r_px ** 2)
        if self.area < self.small_cell_thresh_c * robot_area_px:
            self.size_type = "small"

        # disable planning in very small cells
        if self.area < robot_area_px * 4:  # param
            # print(f"[WARN] cell area is too small: {self.id}")
            self.do_planning = False

    def eval_cell_slimness(self) -> None:
        """ Evaluate cell slimness based on maximum inscribed circle. """
        insc_circle_poly = maximum_inscribed_circle(self.poly)
        insc_circle_center, circle_insc_boundary = insc_circle_poly.coords
        insc_circle_radius = np.linalg.norm(np.array(insc_circle_center) - np.array(circle_insc_boundary))
        if insc_circle_radius < self.robot_r_px * self.slim_cell_thresh_c:
            self.slim_type = "slim"

        # if insc_circle_radius < self.robot_r_px: # todo
        #     print(f"[WARN] cell is too slim: {self.id}")
        #     self.do_planning = False

    def create_cell_points(self):
        """Extract integer coordinate points from cell polygon boundary."""
        if self.poly:
            points = [[int(p[0]), int(p[1])] for p in self.poly.exterior.coords]
            self.points = np.array(points, dtype=np.int32)
        else:
            self.points = []

    def simplify_cell_poly(self, polygon: Polygon, tolerance: float = 1.0) -> Polygon:
        """ Simplify cell polygon to reduce complexity. """
        poly = polygon.simplify(tolerance, preserve_topology=True)
        return poly

    # --------------------------------------------------------------------------
    # setters
    # --------------------------------------------------------------------------

    def add_adjacent_id(self, ind: int):
        """ Add an adjacent cell ID to the list. """
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
        """ Set the starting point of the cell path. """
        self.start_point = point

    def set_end_point(self, point: MyPoint):
        """ Set the ending point of the cell path. """
        self.end_point = point

    # --------------------------------------------------------------------------
    # validation
    # --------------------------------------------------------------------------

    def validate_cell(self) -> bool:
        """ Validate cell integrity. """
        is_valid = True
        if self.area <= 0:
            self.lg.warn(f"Cell has non-positive area: {self.area}")
            is_valid = False
        if not self.poly.is_valid:
            self.lg.warn("Cell polygon is not valid.")
            is_valid = False
        if not self.poly.is_simple:
            self.lg.warn("Cell polygon is not simple.")
            is_valid = False
        if len(self.points) < 3:
            self.lg.warn("Cell polygon has less than 3 points.")
            is_valid = False

        if not is_valid:
            self.do_planning = False
            return False
        return True
