""" contour based coverage planning inside the given cell """
from shapely.affinity import scale
from shapely.geometry import Polygon, MultiPolygon
from scripts.base.logger import MyLogger
from scripts.planning.base_cell import BaseCell
from scripts.base.configs import PlanningConfigs
from scripts.base.my_utils import MyPoint, MyPath


class CellCoveragePathPlanner:
    cell: BaseCell
    path_m: MyPath
    path_px: MyPath
    #
    pcfgs: PlanningConfigs
    resolution: float

    def __init__(self, cell: BaseCell, zone_ind: int = 0, resolution: float = 0.05):
        self.ns = "CellCoverageP-Z" + str(zone_ind)
        self.lg = MyLogger(self.ns)
        self.cell = cell
        self.path_m = None
        self.path_px = None
        #

        # parameters and configurations
        self.pcfgs = PlanningConfigs()
        self.resolution = resolution

        #
        self.coverage_planning()

    def coverage_planning(self) -> None:
        if not self.cell.do_planning:
            self.lg.warn(f"cell {self.cell.id} does not require planning.")
            return
        self.calculate_path_px()
        self.create_path_m()

    def calculate_path_px(self) -> None:
        poly = self.cell.poly
        # scale the polygon to smaller size

        robot_r_px = self.pcfgs.cleaning_r_ov / self.resolution
        robot_d_px = robot_r_px * 2

        scaled_polys = []
        points = []

        # scaled_poly = scale(poly, xfact=0.5, yfact=0.5, origin='center')

        i = 1
        scaled_poly = poly.buffer(-1, join_style=2, resolution=8)  # initial scale
        while not (scaled_poly.is_empty or scaled_poly is None):
            #  if result is MultiPolygon
            if isinstance(scaled_poly, MultiPolygon):
                scaled_poly = scaled_poly.geoms[0]  # take the first polygon
            scaled_polys.append(scaled_poly)
            points.extend(list(scaled_poly.exterior.coords))
            scaled_poly = poly.buffer(-robot_d_px*i, join_style=2, resolution=8)  # scale down the polygon
            i += 1

        # if scaled_poly.is_empty or scaled_poly is None:
        #     self.lg.error(f"Scaled polygon for cell {self.cell.id} is empty or None.")
        #     # raise ValueError(f"Scaled polygon for cell {self.cell.id} is empty or None.")
        #     return

        # create a path from the scaled polygon
        points = [MyPoint(int(p[0]), int(p[1])) for p in points]
        self.path_px = MyPath(points)

    def create_path_m(self) -> None:
        if self.path_px:
            self.path_m = MyPath([self.point_px_to_m(p) for p in self.path_px.points])

    def point_px_to_m(self, p: MyPoint) -> MyPoint:
        return MyPoint(p.x * self.resolution, p.y * self.resolution)
