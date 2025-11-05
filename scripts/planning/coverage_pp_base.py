""" Base class for cell coverage path planner. """
import numpy as np
from scripts.base.logger import MyLogger
from scripts.base.my_utils import MyPoint, MyPath
from scripts.segmentation.cell_processed import PCell
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import PlanningConfig, PipeLineConfig


class CoveragePathPlannerBase:
    """ Base class for cell coverage path planner. """

    def __init__(self, cell: PCell, zmd: ZoneMapData):
        self.lg = MyLogger("CoveragePlannerBase-Z" + str(zmd.ind))

        # zone map data
        self.zmd: ZoneMapData = zmd

        # configurations
        self.dcfgs = PipeLineConfig()
        self.pcfg = PlanningConfig()
        self.resolution: float = zmd.resolution

        # cell
        self.cell: PCell = cell

        # resulted paths
        self.path_m: list[tuple[float, float]] = []   # path in meters
        self.path_px: list[tuple[int, int]] = []      # path in pixels

    def coverage_planning(self) -> None:
        """ Main function to plan the coverage path for the cell.
            To be implemented in child classes.
        """
        raise NotImplementedError("This method should be implemented in child classes.")

    # ==========================================================================

    def create_path_m(self) -> None:
        if self.path_px:
            self.path_m = MyPath([self.point_px_to_m(p) for p in self.path_px.points])

    def path_px_to_m(self, path_px: MyPath) -> MyPath:
        return MyPath([self.point_px_to_m(p) for p in path_px.points])

    def point_px_to_m(self, p: MyPoint) -> MyPoint:
        return MyPoint(p.x * self.resolution, p.y * self.resolution)

    def rotate_path(self, path: MyPath, angle: float, origin: tuple) -> None:
        """ Rotate path by angle (in degrees) around the cell center. """
        return MyPath([self.rotate_point(p, angle, origin) for p in path.points])

    def rotate_point(self, p: MyPoint, angle: float, origin: tuple) -> MyPoint:
        """ Rotate point by angle (in radians) around the origin. """
        ox, oy = origin
        px, py = p.x, p.y

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return MyPoint(int(qx), int(qy))
