""" contour based coverage planning inside the given cell """
import numpy as np
from scripts.base.logger import MyLogger
from scripts.planning.base_cell import BaseCell
from scripts.base.configs import PlanningConfigs
from scripts.base.my_utils import MyPoint, MyPath
# from shapely.geometry import Polygon, Point, LineString


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

    def create_path_px(self) -> None:
        self.path_px = MyPath([self.point_m_to_px(p) for p in self.path_m.points])

    def point_m_to_px(self, p: MyPoint) -> MyPoint:
        return MyPoint(int(p.x / self.resolution), int(p.y / self.resolution))
