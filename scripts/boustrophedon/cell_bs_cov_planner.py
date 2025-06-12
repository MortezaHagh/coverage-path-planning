""" boustrophedon coverage planning inside the given cell """
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scripts.boustrophedon.cell import Cell
from scripts.base.configs import PlanningConfigs
from scripts.base.my_utils import MyPoint, MyPath


class CellCoveragePathPlanner:
    cell: Cell
    poly_m: Polygon
    path_m: MyPath
    path_px: MyPath
    #
    pcfgs: PlanningConfigs
    resolution: float

    def __init__(self, cell: Cell, zone_ind: int = 0, resolution: float = 0.05):
        self.ns = "CellCoverageP-Z" + str(zone_ind)
        self.cell = cell
        self.poly_m = None
        self.path_m = None
        self.path_px = None
        #
        self.xy_but = None
        self.xy_top = None

        # parameters and configurations
        self.pcfgs = PlanningConfigs()
        self.resolution = resolution

        #
        self.coverage_planning()

    def coverage_planning(self) -> None:
        # if not self.eveal_cell():
        #     return
        if not self.cell.do_planning:
            print(f"[{self.ns}]: [WARN] cell {self.cell.id} does not require planning.")
            return

        self.create_cell_polygon_in_meter()
        succ = self.find_top_but_hit_points()
        if not succ:
            return
        self.calculate_path()
        self.create_path_px()

    # def eveal_cell(self):
    #     min_area = 0.5 * np.pi * self.pcfgs.cleaning_r**2
    #     if (self.cell.area*self.resolution**2) < min_area:
    #         print(f"[{self.ns}]: [WARN] cell area is too small: {self.cell.id}")
    #         return False
    #     return True

    def create_cell_polygon_in_meter(self) -> None:
        # create cell polygon in meter
        points = self.cell.points
        points_m = [(p[0] * self.resolution, p[1] * self.resolution) for p in points]
        points_m_p = [Point(p) for p in points_m]
        self.poly_m = Polygon(points_m_p)

    def find_top_but_hit_points(self) -> bool:
        # # parameters and configurations
        cleaning_d = 2*self.pcfgs.cleaning_r_ov
        resolution = self.resolution
        cell = self.cell

        # start point - top left
        x_min = cell.x_min * resolution
        x_max = cell.x_max * resolution
        y_min = cell.y_min * resolution
        y_max = cell.y_max * resolution

        # width
        s_w_m = cell.width * resolution
        # shift
        d_shift = resolution * 5

        # #
        # print(cell.id, "l", cell.shift_le, "r", cell.shift_re)
        # s_w_m = w_m - cleaning_d * (cell.shift_le + cell.shift_re)
        # if cell.shift_le:
        #     x_max -= cleaning_d
        # if cell.shift_re:
        #     x_min += cleaning_d

        #
        if s_w_m < cleaning_d/3:  # todo
            if cell.shift_le or cell.shift_re:
                return False
            xx = [(x_min + x_max)/2]
        elif s_w_m < cleaning_d*1.1:
            xx = [x_min+d_shift, x_max-d_shift]
        else:
            ww = s_w_m - cleaning_d
            n_r = int(np.ceil(ww / (cleaning_d)))
            d_c = s_w_m / (n_r+1)
            #
            xl = x_min + d_shift
            xr = x_max - d_shift
            xx = [x_min + d_c*i for i in range(1, n_r+1)]
            xx.insert(0, xl)
            xx.append(xr)

        # check shifting
        if cell.shift_le:
            xx.pop(0)
        if cell.shift_re:
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
            print(f"[{self.ns}]: [WARN] no hitting points")
            return False
        self.xy_but = xy_but
        self.xy_top = xy_top
        return True

    def get_intersection_points(self, y_check: float, top_point: Point, polygon: Polygon) -> tuple[Point, Point]:
        vertical_line = LineString([top_point, (top_point.x, y_check + 1)])

        # Intersect that vertical line with the polygon
        intersection = vertical_line.intersection(polygon)

        #
        if intersection.is_empty:
            print(f"[{self.ns}]: [WARN] no intersection")
            return None
        elif intersection.geom_type == 'Point':
            print(f"[{self.ns}]: [WARN] only one intersection point: {intersection.x}, {intersection.y}")
            return None
        elif intersection.geom_type == 'LineString':
            xy = intersection.coords.xy
            xx = list(xy[0])
            yy = list(xy[1])
            if len(xx) != 2:
                print(f"[{self.ns}]: [WARN] number of intersection points: {len(xx)}")
                return None
        else:
            print("[{self.ns}]: [WARN] No valid intersection below the point.")
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
