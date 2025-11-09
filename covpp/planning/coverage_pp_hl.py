"""Coverage path planning handler module for managing cell-level path generation.

This module implements the Handler pattern to dynamically load and execute
different coverage planning algorithms based on configuration.
"""

from importlib.metadata import entry_points
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.segmentation.cell_processed import PCell
from covpp.district.zone_map_data import ZoneMapData
from covpp.planning.coverage_pp_base import CoveragePathPlannerBase


class CoveragePPHandler:
    """High-level interface for coverage path planning across all zone cells.

    This handler dynamically loads the appropriate coverage planning algorithm
    based on pipeline configuration and applies it to all cells in the zone.

    Args:
        zone_map_data: Zone map data object.

    Attributes:
        zmd: Zone map data object.
        cells: List of cells to plan paths for.
        all_cell_covp_planners: List of planner instances for each cell.
        lg: Logger instance.

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

        # list of cell coverage planners for cells (loaded in the main planner)
        self.all_cell_covp_planners: list[CoveragePathPlannerBase] = []

    def plan(self, cells: list[PCell]):
        """Execute coverage path planning for all cells.

        Args:
            cells: List of processed cells from segmentation.
        """

        self.cells = cells

        # coverage path planning for all cells
        self.run_covpp_method()

    def run_covpp_method(self) -> None:
        """Apply coverage path planning algorithm to all cells in the zone.

        This method loads the configured planning algorithm and creates a
        planner instance for each cell. Successfully planned paths are stored
        in the cell objects (path_m and path_px attributes).

        Raises:
            ValueError: If coverage planning method is not recognized.
        """

        self.lg.info2("start coverage path planning for cells")

        # create coverage path planner
        # coverage path planning method from configs
        dcfg = PipeLineConfig()
        covpp_method = dcfg.covpp_method
        planners = entry_points(group="planners")
        if covpp_method not in planners.names:
            raise ValueError(f"Coverage planning method '{covpp_method}' not recognized.")
        planner_cls: CoveragePathPlannerBase = planners[covpp_method].load()

        # plan for all cells
        cell: PCell
        for cell in self.cells:
            cell_covp_planner = planner_cls(cell=cell, zmd=self.zmd)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                cell.set_path_m(cell_covp_planner.path_m)
                cell.set_path_px(cell_covp_planner.path_px)
            else:
                cell.do_planning = False
                self.lg.warn(f"Cell {cell.id} has no path after coverage planning.")

    # def zone_covpp(self) -> None:
    #     """ apply coverage path planning for all cells in the zone """
    #     self.lg.info2("start coverage path planning for cells")

    #     cell: PCell
    #     adj_cell: PCell
    #     for cell in self.cells:
    #         cell_covp_planner = CellCoveragePathPlanner(cell=cell, z)
    #         self.all_cell_covp_planners.append(cell_covp_planner)
    #         if cell_covp_planner.path_m:
    #             cell.set_path_m(cell_covp_planner.path_m)
    #             cell.set_path_px(cell_covp_planner.path_px)
    #         else:
    #             # update adjacent cells to remove this cell
    #             cell.do_planning = False
    #             # self.lg.warn(f"Cell {cell.id} has no path after coverage planning.")
    #             # fix adjacents and parent cells
    #             adjs = cell.adjacent_ids[:]
    #             for adj_id in adjs:
    #                 adj_cell = self.cells[adj_id]
    #                 adj_cell.adjacent_ids.remove(cell.id)
    #                 adj_cell.adjacent_ids.extend(cell.adjacent_ids)
    #                 adj_cell.adjacent_ids = list(set(adj_cell.adjacent_ids))  # remove duplicates
    #             cell.adjacent_ids = []
