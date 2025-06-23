import cv2 as cv
import numpy as np
from scripts.base.my_utils import get_n_colors_cv
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.planning.boustrophedon.cell_bs_cov_planner import CellCoveragePathPlanner
from scripts.planning.boustrophedon.decomposition_interface import DecompositionInterface
from scripts.planning.connect_paths import ConnectPaths


class CoveragePP(CoveragePPBase):
    """  Boustrophedon coverage path planning."""
    colors: list
    dec_interface: DecompositionInterface
    all_cell_covp_planners: list[CellCoveragePathPlanner]

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "BoustCoveragePP" + "-Z" + str(self.zmd.ind)
        self.lg.name = self.ns
        self.lg.info1("Step: coverage path planning")

        # configs
        self.colors = []

        # for boustrophedon decomposition
        self.dec_interface = None

        # all cells coverage path planners
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

        # run decomposition and coverage planning
        self.plan()

    def coverage_planning(self) -> None:
        self.decompose_map()
        self.zone_covpp()
        self.bs_post_planning()

    def decompose_map(self) -> None:
        self.lg.info2("start boustrophedon decomposition")
        self.dec_interface = DecompositionInterface(self.map_polygons, self.zmd.reslolution)

    def zone_covpp(self) -> None:
        self.lg.info2("start coverage path planning for cells")
        for mell in self.dec_interface.decomposed.smcells:
            cell_covp_planner = CellCoveragePathPlanner(cell=mell, zone_ind=self.zmd.ind, resolution=self.resolution)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                self.paths_m.append(cell_covp_planner.path_m)
                self.paths_px.append(cell_covp_planner.path_px)

    def bs_post_planning(self) -> None:
        n = self.dec_interface.decomposed.n_mcells
        self.colors = get_n_colors_cv(n)
        self.save_decomposition_plot()
        #
        self.add_cells_to_map_planning()
        self.add_cell_paths_to_map_planning()
        #
        self.connect_paths()

    # ==================================================================================================================

    def connect_paths(self) -> None:
        cp = ConnectPaths(self.zmd, self.paths_px)
        self.c_paths_px = cp.c_paths_px
        self.c_paths_m = [self.path_px_to_m(path_px) for path_px in self.c_paths_px]

    def save_decomposition_plot(self) -> None:
        if not self.dcfgs.save_decomposition_plot:
            return
        self.dec_interface.decomposed.save_figs(self.zmd.result_path)

    def add_cells_to_map_planning(self) -> None:
        for i, cell in enumerate(self.dec_interface.decomposed.cells_groups):
            pp = cell.merged_cell.points
            pp = np.array(pp, dtype=np.int32)  # Ensure pp is a numpy array of integers
            map_p = cv.polylines(self.zmd.map_planning, [pp], isClosed=True, color=(100, 100, 100), thickness=2)
            map_p_fill = self.zmd.map_planning.copy()
            cv.fillPoly(map_p_fill, [pp], color=self.colors[i])
            alpha = 0.1
            self.zmd.map_planning = cv.addWeighted(map_p_fill, alpha, map_p, 1 - alpha, 0)

    def add_cell_paths_to_map_planning(self) -> None:
        map_planning = self.zmd.map_planning.copy()
        for i, cell_covp_planner in enumerate(self.all_cell_covp_planners):
            path_px = cell_covp_planner.path_px
            if path_px:
                pp = np.array(path_px.xy, dtype=np.int32)  # Ensure pp is a numpy array of integers
                map_planning = cv.polylines(map_planning, [pp], isClosed=False, color=self.colors[i], thickness=2)
        self.zmd.map_planning = map_planning
