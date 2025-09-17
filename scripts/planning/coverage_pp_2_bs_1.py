import cv2 as cv
import numpy as np
from scripts.segmentation.cell_base import CellBase
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_0_base import CoveragePPBase
from scripts.planning.boustrophedon.cell_cov_planner import CellCoveragePathPlanner
# from scripts.planning.connect_paths import ConnectPaths
# from scripts.planning.cells_graph_1 import CellsGraph


class CoveragePP(CoveragePPBase):
    """  Boustrophedon coverage path planning. """

    def __init__(self, zone_map_data: ZoneMapData, cells: list[CellBase]):
        super().__init__(zone_map_data, cells)
        self.ns: str = "BoustCoveragePP" + "-Z" + str(self.zmd.ind)
        self.lg.name = self.ns

        # main
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

    def zone_covpp(self) -> None:
        """ apply coverage path planning for all cells in the zone """
        self.lg.info2("start coverage path planning for cells")

        cell: CellBase
        adj_cell: CellBase
        for cell in self.cells:
            cell_covp_planner = CellCoveragePathPlanner(cell=cell, zone_ind=self.zmd.ind, resolution=self.zmd.resolution)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                cell.set_path_m(cell_covp_planner.path_m)
                cell.set_path_px(cell_covp_planner.path_px)
            else:
                # update adjacent cells to remove this cell
                cell.do_planning = False
                # self.lg.warn(f"Cell {cell.id} has no path after coverage planning.")
                # fix adjacents and parent cells
                adjs = cell.adjacent_ids[:]
                for adj_id in adjs:
                    adj_cell = self.cells[adj_id]
                    adj_cell.adjacent_ids.remove(cell.id)
                    adj_cell.adjacent_ids.extend(cell.adjacent_ids)
                    adj_cell.adjacent_ids = list(set(adj_cell.adjacent_ids))  # remove duplicates
                cell.adjacent_ids = []
