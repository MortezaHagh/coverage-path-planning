from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfigs
from scripts.segmentation.cell_base import CellBase
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_0_base import CoveragePPBase


class CoveragePathPlanner:
    def __init__(self, zone_map_data: ZoneMapData):
        self.ns: str = 'CoveragePathPlanner'
        self.lg = MyLogger(self.ns)

        # zone map data
        self.zmd: ZoneMapData = zone_map_data

        # coverage path planner for cells
        self.cov_path_planner: CoveragePPBase = None

    def plan(self, cells: list[CellBase]):
        """Run coverage path planning."""

        # coverage path planning method from configs
        dcfg = DistrictConfigs()
        covpp_method = dcfg.covpp_method

        # create coverage path planner
        if covpp_method == "boust":
            from scripts.planning.coverage_pp_2_bs_1 import CoveragePP
        elif covpp_method == "contour":
            from scripts.planning.coverage_pp_3_contour import CoveragePP
        else:
            raise ValueError(f"Unknown coverage path planning method: {covpp_method}. ")
        self.cov_path_planner = CoveragePP(self.zmd, cells)

        # plan
        self.cov_path_planner.plan()
