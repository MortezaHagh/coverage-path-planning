"""Coverage path planning handler module for managing cell-level path generation.

This module implements the Handler pattern to dynamically load and execute
different coverage planning algorithms based on configuration.
"""

from importlib.metadata import entry_points
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.segmentation.cell_planning import PCell
from covpp.district.zone_map_data import ZoneMapData
from covpp.coverage_planning.coverage_pp_base import CoveragePathPlannerBase


class CoveragePPHandler:
    """High-level interface for coverage path planning across all zone cells.

    This handler dynamically loads the appropriate coverage planning algorithm
    based on pipeline configuration and applies it to all cells in the zone.

    Args:
        zone_map_data (ZoneMapData): Zone map data object.

    Attributes:
        zmd (ZoneMapData): Zone map data object.
        cells (list[PCell]): List of cells to plan paths for.
        all_cell_covp_planners (list[CoveragePathPlannerBase]): List of planner instances for each cell.

    Result:
        After planning, path_m and path_px are set for each cell.

    Supported Methods:
        - "boust": Boustrophedon (back-and-forth) coverage
        - "contour": Contour-following coverage
        - Custom methods can be added by extending CoveragePathPlannerBase
    """

    def __init__(self, zone_map_data: ZoneMapData):
        self.lg = MyLogger('CoveragePPHandler')

        # zone map data
        self.zmd: ZoneMapData = zone_map_data

        # input cells from segmentation
        self.cells: list[PCell] = []

        # # list of cell coverage planners for cells (loaded in the main planner)
        # self.all_cell_covp_planners: list[CoveragePathPlannerBase] = []

    def run(self, cells: list[PCell]):
        """Execute coverage path planning for all cells.

        Args:
            cells: List of processed cells from segmentation.
        """
        self.cells = cells
        self.run_covpp_method()

    def run_covpp_method(self) -> None:
        """Apply coverage path planning algorithm to all cells in the zone.

        This method loads the configured planning algorithm and creates a
        planner instance for each cell. Successfully planned paths are stored
        in the cell objects (path_m and path_px attributes).
        """

        self.lg.info2("start coverage path planning for cells")

        # ----------------------------------------------------------------------
        # create coverage path planner
        # coverage path planning method from configs
        covpp_method = PipeLineConfig.covpp_method
        planners = entry_points(group="coverage_planners")
        if covpp_method not in planners.names:
            raise ValueError(f"Coverage planning method '{covpp_method}' not recognized.")
        planner_cls = planners[covpp_method].load()
        # ----------------------------------------------------------------------

        # plan for all cells
        cell: PCell
        cell_covp_planner: CoveragePathPlannerBase

        for cell in self.cells:

            if not cell.do_planning:
                self.lg.info(f"Skipping cell {cell.id} as it is marked not for planning.")
                continue

            cell_covp_planner = planner_cls(cell=cell, zmd=self.zmd)
            cell_covp_planner.run()
            if cell_covp_planner.path_m:
                cell.set_path_m(cell_covp_planner.path_m)
                cell.set_path_px(cell_covp_planner.path_px)
            else:
                cell.do_planning = False
                self.lg.warn(f"Cell {cell.id} has no path after coverage planning.")
                self.lg.warn(f"Cell {cell.id} set do_planning to False.")
