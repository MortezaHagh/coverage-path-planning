import cv2 as cv
import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.segmentation.cell_base import CellBase
from scripts.planning.coverage_pp_0_base import CoveragePPBase
from scripts.planning.contour.cell_cov_planner import CellCoveragePathPlanner
# from scripts.planning.connect_paths import ConnectPaths
# from scripts.planning.cells_graph_1 import CellsGraph


class CoveragePP(CoveragePPBase):
    """  contour based coverage path planning. """

    def __init__(self, zone_map_data: ZoneMapData, cells: list[CellBase]):
        super().__init__(zone_map_data, cells)
        self.ns: str = "ContourCoveragePP" + "-Z" + str(self.zmd.ind)
        self.lg.name = self.ns

        # main
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

    def zone_covpp(self) -> None:
        """ apply coverage path planning for all cells in the zone """
        self.lg.info2("start coverage path planning for cells")

        cell: CellBase
        for cell in self.cells:
            cell_covp_planner = CellCoveragePathPlanner(cell=cell, zmd=self.zmd, zone_ind=self.zmd.ind)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                cell.set_path_m(cell_covp_planner.path_m)
                cell.set_path_px(cell_covp_planner.path_px)
            else:
                cell.do_planning = False
                self.lg.warn(f"Cell {cell.id} has no path after coverage planning.")
