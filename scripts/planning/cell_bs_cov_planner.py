""" boustrophedon coverage planning inside the given cell """
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scripts.decomposition.cell import Cell
from scripts.base.configs import SpaceConfigs, PlanningConfigs


class CellCoveragePP:
    cell: Cell
    poly_m: Polygon
    path_m: list[tuple[float, float]] = []
    path_px: list[tuple[int, int]] = []

    def __init__(self, cell: Cell):
        self.ns = "CellCoveragePP"
        self.cell = cell
        self.poly_m = None
        self.path_m = []
        self.path_px = []
        #
        self.xy_but = None
        self.xy_top = None

        # parameters and configurations
        self.scfgs = SpaceConfigs()
        self.pcfgs = PlanningConfigs()
        self.resolution = 0.05

        #
        self.coverage_planning()

    def coverage_planning(self):
        self.create_cell_polygon_in_meter()
        succ = self.find_top_but_hit_points()
        if not succ:
            return
        self.calculate_path()
        self.create_path_px()

    def calculate_path(self):
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
        self.path_m = points

    def create_cell_polygon_in_meter(self):
        # create cell polygon in meter
        points = self.cell.poly_points
        points_m = [(p[0] * self.resolution, p[1] * self.resolution) for p in points]
        points_m_p = [Point(p) for p in points_m]
        self.poly_m = Polygon(points_m_p)

    def find_top_but_hit_points(self):

        # # parameters and configurations
        # effective cleaning diameter
        cleaning_width = 2*self.scfgs.cleaning_r
        min_overlap_w = self.pcfgs.min_path_overlap
        cleaning_d = cleaning_width - min_overlap_w   # cleaning diameter
        #
        resolution = self.resolution

        # width and height
        w_m = self.cell.width * resolution
        h_m = self.cell.height * resolution

        # margin
        margin = self.pcfgs.margin
        w_m = w_m - 2 * margin
        h_m = h_m - 2 * margin + cleaning_d

        # start point - top left
        x_min = self.cell.x_min * resolution
        x_max = self.cell.x_max * resolution
        y_min = self.cell.y_min * resolution
        y_max = self.cell.y_max * resolution

        #
        n_lines = np.ceil(w_m / cleaning_d)                     # number of lines
        if n_lines == 0:
            return False
        cleaning_d = w_m / n_lines                              # width of each cleaning line

        # find hitting points with the cell polygon
        x_left = x_min + 1.0  # todo
        y_but = y_min - 1.0  # todo
        xx = [x_left+cleaning_d*i for i in range(int(n_lines))]

        #
        if len(xx) == 0:
            return False
        xx[-1] = xx[-1] - 1.0  # todo
        y_hit_check = y_max + 1.0  # todo

        # find hitting points with the cell polygon
        xy_but = []
        xy_top = []
        for x in xx:
            top_point = Point((x, y_but))
            p1, p2 = self.get_intersection_points(y_hit_check, top_point, self.poly_m)
            xy_but.append((p1.x, p1.y))
            xy_top.append((p2.x, p2.y))
        self.xy_but = xy_but
        self.xy_top = xy_top
        return True

    def get_intersection_points(self, y_check: float, top_point: Point, polygon: Polygon):
        vertical_line = LineString([top_point, (top_point.x, y_check + 1)])

        # Intersect that vertical line with the polygon
        intersection = vertical_line.intersection(polygon)

        #
        if intersection.is_empty:
            print(f"[{self.ns}]: no intersection")
            raise ValueError("No intersection found between the line and the polygon.")
        elif intersection.geom_type == 'Point':
            print(f"only one intersection point: {intersection.x}, {intersection.y}")
            raise ValueError("Only one intersection point found.")
        elif intersection.geom_type == 'LineString':
            xy = intersection.coords.xy
            xx = list(xy[0])
            yy = list(xy[1])
            if len(xx) != 2:
                print(f"[{self.ns}]: number of intersection points: {len(xx)}")
                raise ValueError("Invalid number of intersection points found.")
        else:
            print("No valid intersection below the point.")
            raise ValueError("No valid intersection below the point.")
        #
        p1 = Point(xx[0], yy[0])
        p2 = Point(xx[1], yy[1])
        return (p1, p2)

    def create_path_px(self):
        self.path_px = [self.point_m_to_px(p) for p in self.path_m]

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        return (int(p[0] / self.resolution), int(p[1] / self.resolution))
