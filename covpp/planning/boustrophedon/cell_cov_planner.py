"""Boustrophedon coverage path generation for decomposed trapezoidal cells."""
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from covpp.base.logger import MyLogger
from covpp.base.configs import PlanningConfig
from covpp.base.my_utils import MyPoint, MyPath
from covpp.planning.boustrophedon.cell import Cell


class CellCoveragePathPlanner:
    """Generate sweeping coverage paths for individual trapezoidal cells.

    Creates alternating back-and-forth coverage paths within a decomposed cell
    by finding intersection points with horizontal scan lines.

    Planning Process:
        1. Convert cell polygon to meter coordinates
        2. Find hitting points with cell boundary from top and bottom
        3. Connect hitting points in alternating boustrophedon pattern
        4. Convert final path to pixel coordinates
    """

    def __init__(self, cell: Cell, zone_ind: int = 0, resolution: float = 0.05):
        self.lg = MyLogger(f"CellCoverageP-Z{zone_ind}-C{cell.id}")
        #
        self.cell: Cell = cell
        self.path_m: MyPath = None
        self.path_px: MyPath = None
        #
        self.poly_m: Polygon = None
        self.xy_but: list[tuple[float, float]] = None
        self.xy_top: list[tuple[float, float]] = None

        # parameters and configurations
        self.pcfg = PlanningConfig()
        self.resolution: float = resolution

        #
        self.coverage_planning()

    def coverage_planning(self) -> None:
        """Execute boustrophedon path planning for the trapezoidal cell."""
        if not self.cell.do_planning:
            # self.lg.warn(f"--- Boust cell {self.cell.id} does not require planning.")
            return

        if not self.create_cell_polygon_in_meter():
            return

        succ = self.find_top_but_hit_points()
        if not succ:
            return
        self.calculate_path()
        self.create_path_px()

    # def eval_cell(self):
    #     min_area = 0.5 * np.pi * self.pcfg.cleaning_r**2
    #     if (self.cell.area*self.resolution**2) < min_area:
    #         self.lg.warn(f"cell area is too small: {self.cell.id}")
    #         return False
    #     return True

    def create_cell_polygon_in_meter(self) -> bool:
        """Convert cell polygon from pixel to meter coordinates and validate."""
        points = self.cell.points
        points_m = [(p[0] * self.resolution, p[1] * self.resolution) for p in points]
        points_m_p = [Point(p) for p in points_m]
        self.poly_m = Polygon(points_m_p)

        # check validity
        if not self.poly_m.is_valid:
            self.lg.warn(f"Cell polygon (in meter) {self.cell.id} is not valid.")
            fixed_poly = self.poly_m.buffer(0)
            if fixed_poly.is_valid and not fixed_poly.is_empty and fixed_poly.geom_type == 'Polygon':
                self.poly_m = fixed_poly
                self.lg.warn(f"Cell polygon (in meter) {self.cell.id} is fixed.")
            else:
                self.cell.do_planning = False
                return False

        return True

    def find_top_but_hit_points(self) -> bool:
        """
        Find hitting points with the cell polygon from top and bottom.

        Process:
            1. Determine x-coordinates of vertical scan lines based on cell width
               and cleaning distance, adjusting for any shifts.
            2. For each x-coordinate, create a vertical line from above the cell
               downwards and find intersection points with the cell polygon.
            3. Store the bottom and top intersection points for path generation.
        """

        # parameters and configurations
        cleaning_d = 2*self.pcfg.cleaning_r_ov
        resolution = self.resolution
        cell = self.cell

        # start point - top left
        x_min = cell.x_min_p * resolution
        x_max = cell.x_max_p * resolution
        y_min = cell.y_min_p * resolution
        y_max = cell.y_max_p * resolution

        # width in meters
        w_m = cell.width * resolution
        # shift
        d_shift = resolution * 5

        # find xx: the x coordinates of the vertical scan lines
        w_thresh_1 = cleaning_d / 3
        w_thresh_2 = cleaning_d * 1.1
        if w_m < w_thresh_1:
            if cell.shift_l or cell.shift_r:
                return False
            xx = [(x_min + x_max)/2]
        elif w_m < w_thresh_2:
            xx = [x_min+d_shift, x_max-d_shift]
        else:
            ww = w_m - cleaning_d
            n_r = int(np.ceil(ww / (cleaning_d)))
            d_c = w_m / (n_r+1)
            #
            xl = x_min + d_shift
            xr = x_max - d_shift
            xx = [x_min + d_c*i for i in range(1, n_r+1)]
            xx.insert(0, xl)
            xx.append(xr)

        # check shifting: remove first/last if shifted
        if cell.shift_l:
            xx.pop(0)
        if cell.shift_r:
            xx.pop(-1)

        # find hitting points with the cell polygon
        y_but = y_min - d_shift
        y_hit_check = y_max + d_shift
        xy_but = []
        xy_top = []
        for x in xx:
            top_point = Point((x, y_but))
            points = self.get_intersection_points(y_hit_check, top_point, self.poly_m)
            if points:
                p1, p2 = points
                xy_but.append((p1.x, p1.y+d_shift))
                xy_top.append((p2.x, p2.y-d_shift))
        if len(xy_but) < 1:
            self.lg.warn("no hitting points")
            return False
        self.xy_but = xy_but
        self.xy_top = xy_top
        return True

    def get_intersection_points(self, y_check: float, top_point: Point, polygon: Polygon) -> tuple[Point, Point]:
        """ get intersection points of a vertical line below the top_point with the polygon """

        vertical_line = LineString([top_point, (top_point.x, y_check + 1)])

        # Intersect that vertical line with the polygon
        intersection = vertical_line.intersection(polygon)

        #
        if intersection.is_empty:
            self.lg.warn("no intersection")
            return None
        elif intersection.geom_type == 'Point':
            self.lg.warn(f"only one intersection point: {intersection.x}, {intersection.y}")
            return None
        elif intersection.geom_type == 'LineString':
            xy = intersection.coords.xy
            xx = list(xy[0])
            yy = list(xy[1])
            if len(xx) != 2:
                self.lg.warn(f"number of intersection points: {len(xx)}")
                return None
        else:
            self.lg.warn("No valid intersection below the point.")
            return None
        #
        p1 = Point(xx[0], yy[0])
        p2 = Point(xx[1], yy[1])
        return (p1, p2)

    def calculate_path(self) -> None:
        points_d = self.xy_but
        points_u = self.xy_top
        points = [points_d.pop(0)]
        i_d, i_u = 0, 0
        turn = 'u'  # start with down

        while i_d < len(points_d) or i_u < len(points_u):
            if turn == 'd' and i_d < len(points_d):
                count = min(2, len(points_d) - i_d)
                points.extend(points_d[i_d:i_d+count])
                i_d += count
                turn = 'u'
            elif turn == 'u' and i_u < len(points_u):
                count = min(2, len(points_u) - i_u)
                points.extend(points_u[i_u:i_u+count])
                i_u += count
                turn = 'd'
            else:
                break  # no more points to add

        #
        self.path_m = MyPath([MyPoint(p[0], p[1]) for p in points])
        # self.path_m = points

    def create_path_px(self) -> None:
        self.path_px = MyPath([self.point_m_to_px(p) for p in self.path_m.points])

    def point_m_to_px(self, p: MyPoint) -> MyPoint:
        return MyPoint(int(p.x / self.resolution), int(p.y / self.resolution))
