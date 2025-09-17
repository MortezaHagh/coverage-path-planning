

from scripts.base.logger import MyLogger
from scripts.base.configs import PlanningConfigs
from scripts.segmentation.cell_base import CellBase
from scripts.district.zone_map_data import ZoneMapData


class CoveragePathPlannerBase:
    """ Base class for cell coverage path planner. """

    def __init__(self, cell: CellBase, zmd: ZoneMapData):
        self.ns = "CellCoveragePlannerBase-Z" + str(zmd.ind)
        self.lg = MyLogger(self.ns)

        # configurations
        self.pcfgs = PlanningConfigs()
        self.resolution: float = zmd.resolution

        # cell
        self.cell: CellBase = cell
        self.zmd: ZoneMapData = zmd
        self.zone_ind: int = zmd.ind

        # path
        self.path_m: list[tuple[float, float]] = []   # path in meters
        self.path_px: list[tuple[int, int]] = []      # path in pixels
