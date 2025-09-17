from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfigs
from scripts.segmentation.cell_base import CellBase
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_base import CoveragePathPlannerBase


class CoveragePathPlanHandler:
    def __init__(self, zone_map_data: ZoneMapData):
        self.ns: str = 'CoveragePathPlanHandler'
        self.lg = MyLogger(self.ns)

        # zone map data
        self.zmd: ZoneMapData = zone_map_data

        # input cells from segmentation
        self.cells: list[CellBase] = []

        # list of cell coverage planners for cells (loaded in the main planner)
        self.all_cell_covp_planners: list[CoveragePathPlannerBase] = []

    def plan(self, cells: list[CellBase]):
        """Run coverage path planning."""

        self.cells = cells

        # coverage path planning for all cells
        self.all_cells_covpp()

    def all_cells_covpp(self) -> None:
        """ apply coverage path planning for all cells in the zone """

        self.lg.info2("start coverage path planning for cells")

        # coverage path planning method from configs
        dcfg = DistrictConfigs()
        covpp_method = dcfg.covpp_method

        # create coverage path planner
        if covpp_method == "boust":
            from scripts.planning.boustrophedon.coverage_pp_boust import CoveragePathPlanner
        elif covpp_method == "contour":
            from scripts.planning.contour.coverage_pp_contour import CoveragePathPlanner
        else:
            raise ValueError(f"Unknown coverage path planning method: {covpp_method}. ")

        #
        cell: CellBase
        for cell in self.cells:
            cell_covp_planner = CoveragePathPlanner(cell=cell, zmd=self.zmd)
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

    #     cell: CellBase
    #     adj_cell: CellBase
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
