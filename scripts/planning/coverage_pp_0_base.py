

from scripts.base.logger import MyLogger
from scripts.base.configs import PlanningConfigs
from scripts.segmentation.cell_base import CellBase
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.my_utils import get_n_colors_cv


class CoveragePPBase:
    """ Base class for coverage path planning. """

    def __init__(self, zone_map_data: ZoneMapData, cells: list[CellBase]):
        self.ns: str = "CoveragePPBase"
        self.lg = MyLogger(self.ns)

        # zone map data
        self.zmd: ZoneMapData = zone_map_data
        self.map_polygons: MapPolygons = MapPolygons(self.zmd)

        # input cells from segmentation
        self.cells: list[CellBase] = cells

        # parameters and configurations # todo
        self.pcfgs: PlanningConfigs = PlanningConfigs()

        # list of cell coverage planners for cells (loaded in the main planner)
        self.all_cell_covp_planners: list = []

    def plan(self) -> None:
        self.zone_covpp()       # method specific

    # Planning =========================================================================================================

    def zone_covpp(self) -> None:
        """ apply coverage path planning for all cells in the zone. 

        Virtual method to be implemented by subclasses. 
        """
        raise NotImplementedError("zone_covpp() must be implemented by subclasses.")
