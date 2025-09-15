import cv2 as cv
import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.planning.boustrophedon.cell_cov_planner import CellCoveragePathPlanner
from scripts.planning.boustrophedon.decomposition_interface import DecompositionInterface
# from scripts.planning.connect_paths import ConnectPaths
# from scripts.planning.cells_graph_1 import CellsGraph


class CoveragePP(CoveragePPBase):
    """  Boustrophedon coverage path planning. """

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "BoustCoveragePP" + "-Z" + str(self.zmd.ind)
        self.lg.name = self.ns
        self.lg.info1("Step: coverage path planning")

        # main
        self.dec_interface: DecompositionInterface = None                   # boustrophedon decomposition interface
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

    def decompose_map(self) -> None:
        self.lg.info2("start boustrophedon decomposition")
        self.dec_interface = DecompositionInterface(self.zmd, self.map_polygons)

    def zone_covpp(self) -> None:
        self.lg.info2("start coverage path planning for cells")
        for mcell in self.dec_interface.mcells:
            cell_covp_planner = CellCoveragePathPlanner(cell=mcell, zone_ind=self.zmd.ind, resolution=self.resolution)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                mcell.set_path_m(cell_covp_planner.path_m)
                mcell.set_path_px(cell_covp_planner.path_px)
            else:
                mcell.do_planning = False
                # self.lg.warn(f"Cell {mcell.id} has no path after coverage planning.")
                # fix adjacents and parent cells
                adjs = mcell.adjacent_ids[:]
                for adj_id in adjs:
                    adj_cell = self.dec_interface.mcells[adj_id]
                    adj_cell.adjacent_ids.remove(mcell.id)
                    adj_cell.adjacent_ids.extend(mcell.adjacent_ids)
                    adj_cell.adjacent_ids = list(set(adj_cell.adjacent_ids))  # remove duplicates
                mcell.adjacent_ids = []

    # plot and save ====================================================================================================

    def add_cell_paths_to_map_planning(self) -> None:
        map_planning = self.zmd.map_planning.copy()
        for i, path_px in enumerate(self.paths_px):
            if path_px:
                pp = np.array(path_px.xy, dtype=np.int32)  # Ensure pp is a numpy array of integers
                map_planning = cv.polylines(map_planning, [pp], isClosed=False, color=self.colors[i], thickness=2)
        self.zmd.map_planning = map_planning
