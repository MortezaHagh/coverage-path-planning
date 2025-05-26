import os
import json
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scripts.base.configs import DistrictConfgs
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.boustrophedon.cell_bs_cov_planner import CellCoveragePathPlanner
from scripts.boustrophedon.decomposition_interface import DecompositionInterface
from scripts.base.my_utils import get_n_colors_cv, get_n_colors_plt


class CoveragePP(CoveragePPBase):
    """  BoustCoveragePP  """

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "BoustCoveragePP" + "-Z" + str(self.zmd.ind)
        print(f" >>>>> [{self.ns}] Step: coverage path planning")

        # configs
        self.dcfgs = DistrictConfgs()
        self.colors = []

        # all cells coverage path planners
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

        # run decomposition and coverage planning
        self.plan()

    def coverage_planning(self):
        self.decompose_map()
        self.zone_covpp()
        self.save_data_figs()

    def decompose_map(self):
        print(f"[{self.ns}]: start decomposition")
        self.dec_interface = DecompositionInterface(self.zmd)

    def zone_covpp(self):
        print(f"[{self.ns}]: start coverage path planning for cells")
        for mell in self.dec_interface.decomposed.smcells:
            cell_covp_planner = CellCoveragePathPlanner(cell=mell, zone_ind=self.zmd.ind, resolution=self.resolution)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                self.paths_m.append(cell_covp_planner.path_m)
                self.paths_px.append(cell_covp_planner.path_px)

    def save_data_figs(self):
        n = self.dec_interface.decomposed.n_mcells
        self.colors = get_n_colors_cv(n)
        self.save_decomposition_plot()
        self.add_cells_to_map_planning()
        self.add_cell_paths_to_map_planning()
        self.plot_paths()
        self.save_data()

    # ==================================================================================================================

    def save_decomposition_plot(self):
        if not self.dcfgs.save_decomposition_plot:
            return
        ns = self.dcfgs.covpp_method
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

    # ==========================================================================

    def connect_paths(self):
        start_end_points = []
        for path in self.paths_px:
            start_end_points.append((path[0], path[-1]))
        return start_end_points

    def add_cell_paths_to_map_planning(self):
        map_planning = self.zmd.map_planning.copy()
        for i, cell_covp_planner in enumerate(self.all_cell_covp_planners):
            path_px = cell_covp_planner.path_px
            if path_px:
                pp = np.array(path_px, dtype=np.int32)  # Ensure pp is a numpy array of integers
                map_planning = cv.polylines(map_planning, [pp], isClosed=False, color=self.colors[i], thickness=2)
        self.zmd.map_planning = map_planning

    def plot_paths(self):
        fig, ax = plt.subplots(1, 1)
        self.plot_coverage_paths(ax)
        self.plot_connections(ax)

        #
        ns = self.dcfgs.covpp_method
        boust_path = os.path.join(self.zmd.path, ns)
        seq_method = self.dcfgs.cell_sequence_method
        fig_path = os.path.join(boust_path, f"paths_{seq_method}.png")
        fig.savefig(fig_path, format="png", dpi=300)
        # plt.show()

    def plot_connections(self, ax):
        start_end_points = self.connect_paths()
        ends = [end for _, end in start_end_points]
        starts = [start for start, _ in start_end_points]
        ends = np.array(ends, dtype=np.int32)
        starts = np.array(starts, dtype=np.int32)
        # fig, ax = plt.subplots(1, 1)
        for start, end in zip(starts[1:], ends[:-1]):
            ax.annotate('', xy=(start), xytext=(end), arrowprops=dict(arrowstyle='->,head_width=0.4', color='gray', lw=1.5, ls="--"))
            # ax.plot([end[0], start[0]], [end[1], start[1]], 'k--', linewidth=0.5)

    def plot_coverage_paths(self, ax):
        n = self.dec_interface.decomposed.n_mcells
        colors = get_n_colors_plt(n)
        # fig, ax = plt.subplots(1, 1)
        for obs in self.dec_interface.decomposed.obstacles:
            x, y = obs.poly.exterior.xy
            ax.plot(x, y, color="gray", linewidth=1)

        for i, cell_covp_planner in enumerate(self.all_cell_covp_planners):
            path_px = cell_covp_planner.path_px
            if len(path_px) < 1:
                continue
            pp = np.array(path_px, dtype=np.int32)
            ax.plot(pp[:, 0], pp[:, 1], color=colors[i], linewidth=2, linestyle='-')
            # start and end points
            ax.plot(pp[0, 0], pp[0, 1], 'o', color=colors[i], markersize=10)
            ax.text(pp[0, 0], pp[0, 1], f'S{i}', fontsize=9, color='k')
            ax.plot(pp[-1, 0], pp[-1, 1], 's', color=colors[i], markersize=10)
            ax.text(pp[-1, 0], pp[-1, 1], f'E{i}', fontsize=9, color='k')

    def save_data(self):
        len_px = self.dec_interface.decomposed.smcells_seq_length_px
        len_m = len_px * self.resolution
        data = {"centers_seq_length": round(len_m, 2)}
        with open(self.zmd.result_file, 'w') as f:
            json.dump(data, f, indent=4)
            f.write("\n")
