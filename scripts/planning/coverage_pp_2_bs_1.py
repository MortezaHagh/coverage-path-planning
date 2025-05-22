import os
import cv2 as cv
import numpy as np
from scripts.base.configs import DistrictConfgs  # , SpaceConfigs
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.boustrophedon.cell_bs_cov_planner import CellCoveragePathPlanner
from scripts.boustrophedon.decomposition_interface import DecompositionInterface
from scripts.base.my_utils import get_n_colors_bgr


class CoveragePP(CoveragePPBase):
    """  DecCoveragePP  """

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "DecCoveragePP" + "-Z" + str(self.zmd.ind)
        print(f"[{self.ns}]: >>>>> Step: coverage path planning")

        # configs
        self.dcfg = DistrictConfgs()
        self.colors = []

        # all cells coverage path planners
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

        # run decomposition and coverage planning
        self.plan()

    def coverage_planning(self):
        self.decompose_map()
        self.save_decomposition_plot()

        #
        n = self.dec_interface.decomposed.n_mcells
        self.colors = get_n_colors_bgr(n)
        # self.colors = [(int(np.random.randint(0, cm)), int(np.random.randint(0, cm)), int(np.random.randint(0, cm))) for _ in range(n)]
        self.add_cells_to_map_planning()
        #
        self.coverage_planning_l()

    # =================================================================================================================

    def decompose_map(self):
        print(f"[{self.ns}]: start decomposition")
        self.dec_interface = DecompositionInterface(self.zmd)

    def save_decomposition_plot(self):
        if not self.dcfg.save_decomposition_plot:
            return
        ns = self.dcfg.covpp_method
        boust_path = os.path.join(self.zmd.path, ns)  # boustrophedon
        os.makedirs(boust_path, exist_ok=True)
        self.dec_interface.decomposed.save_figs(boust_path)

    def add_cells_to_map_planning(self):
        for i, cell in enumerate(self.dec_interface.decomposed.cells_groups):
            pp = cell.merged_cell.points
            pp = np.array(pp, dtype=np.int32)  # Ensure pp is a numpy array of integers
            map_p = cv.polylines(self.zmd.map_planning, [pp], isClosed=True, color=(100, 100, 100), thickness=2)
            map_p_fill = self.zmd.map_planning.copy()
            cv.fillPoly(map_p_fill, [pp], color=self.colors[i])
            alpha = 0.1
            self.zmd.map_planning = cv.addWeighted(map_p_fill, alpha, map_p, 1 - alpha, 0)

    # =================================================================================================================

    def coverage_planning_l(self):
        print(f"[{self.ns}]: start coverage path planning")
        self.zone_covpp()
        self.add_cell_paths_to_map_planning()

    def zone_covpp(self):
        for cellg in self.dec_interface.decomposed.cells_groups:
            cell_covp_planner = CellCoveragePathPlanner(cell=cellg.merged_cell, zone_ind=self.zmd.ind, resolution=self.resolution)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                self.paths_m.append(cell_covp_planner.path_m)
                self.paths_px.append(cell_covp_planner.path_px)

    def add_cell_paths_to_map_planning(self):
        robot_m = self.zmd.map_planning.copy()  # np.ones_like(self.zmd.map_planning) * 255
        map_planning = self.zmd.map_planning.copy()
        for i, cell_covp_planner in enumerate(self.all_cell_covp_planners):
            path_px = cell_covp_planner.path_px
            if path_px:
                pp = np.array(path_px, dtype=np.int32)  # Ensure pp is a numpy array of integers
                map_planning = cv.polylines(map_planning, [pp], isClosed=False, color=self.colors[i], thickness=2)
                #
                radius_px = int(np.ceil(self.scfgs.cleaning_r / self.zmd.orig_map_data.up_resolution))
                robot_m = cv.circle(robot_m, pp[0], radius_px, [40, 40, 40], -1)
                if len(pp) > 3:
                    robot_m = cv.circle(robot_m, pp[3], radius_px, [40, 40, 40], -1)
        #
        alpha = 0.8
        self.zmd.map_planning = cv.addWeighted(map_planning, alpha, robot_m, 1 - alpha, 0)
