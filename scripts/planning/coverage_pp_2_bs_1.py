import os
import json
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scripts.base.my_utils import MyPath  # , MyPoint
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.boustrophedon.cell_bs_cov_planner import CellCoveragePathPlanner
from scripts.boustrophedon.decomposition_interface import DecompositionInterface
from scripts.base.my_utils import get_n_colors_cv, get_n_colors_plt
from scripts.planning.connect_paths import ConnectPaths  # , get_start_end_points todo


class CoveragePP(CoveragePPBase):
    """  Boustrophedon coverage path planning."""
    colors: list
    c_paths_px: list[MyPath]
    c_paths_m: list[MyPath]
    dec_interface: DecompositionInterface
    all_cell_covp_planners: list[CellCoveragePathPlanner]

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "BoustCoveragePP" + "-Z" + str(self.zmd.ind)
        print(f" >>>>> [{self.ns}] Step: coverage path planning")

        # configs
        self.colors = []

        # connection paths between cells
        self.c_paths_px = []
        self.c_paths_m = []

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
        print(f"[{self.ns}]: start decomposition")
        self.dec_interface = DecompositionInterface(self.zmd)

    def zone_covpp(self) -> None:
        print(f"[{self.ns}]: start coverage path planning for cells")
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
        self.plot_paths_px()
        self.save_data()
        # self.plot_paths_m()
        #

    # ==================================================================================================================

    def connect_paths(self) -> None:
        cp = ConnectPaths(self.zmd, self.paths_px)
        self.c_paths_px = cp.c_paths_px
        self.c_paths_m = self.path_px_to_m(self.c_paths_px)

    # ==================================================================================================================

    def save_decomposition_plot(self) -> None:
        if not self.dcfgs.save_decomposition_plot:
            return
        ns = self.dcfgs.covpp_method
        boust_path = os.path.join(self.zmd.path, ns)  # boustrophedon
        os.makedirs(boust_path, exist_ok=True)
        self.dec_interface.decomposed.save_figs(boust_path)

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

    # ==================================================================================================================

    def plot_paths_px(self) -> None:
        """ Plot the calculated coverage paths and connections between them.
        """
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        #
        self.plot_coverage_paths(ax)
        self.plot_connection_paths(ax)
        # self.plot_connections_arrow(ax)

        #
        ns = self.dcfgs.covpp_method
        boust_path = os.path.join(self.zmd.path, ns)
        os.makedirs(boust_path, exist_ok=True)
        seq_method = self.dcfgs.cell_sequence_method
        fig_path = os.path.join(boust_path, f"paths_{seq_method}.png")
        fig.savefig(fig_path, format="png", dpi=300)
        # plt.show()

    def plot_coverage_paths(self, ax):
        n = self.dec_interface.decomposed.n_mcells
        colors = get_n_colors_plt(n)
        # fig, ax = plt.subplots(1, 1)
        for obs in self.dec_interface.decomposed.obstacles:
            x, y = obs.poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=1)

        for i, cell_covp_planner in enumerate(self.all_cell_covp_planners):
            path = cell_covp_planner.path_px
            if path is None:
                continue
            pp = np.array(path.xy, dtype=np.int32)
            ax.plot(pp[:, 0], pp[:, 1], color=colors[i], linewidth=2, linestyle='-')
            # start and end points
            ax.plot(pp[0, 0], pp[0, 1], 'o', color=colors[i], markersize=10)
            ax.text(pp[0, 0], pp[0, 1], f'S{i}', fontsize=9, color='k')
            ax.plot(pp[-1, 0], pp[-1, 1], 's', color=colors[i], markersize=10)
            ax.text(pp[-1, 0], pp[-1, 1], f'E{i}', fontsize=9, color='k')

    def plot_connection_paths(self, ax):
        for c_path in self.c_paths_px:
            if len(c_path.xy) < 2:
                continue
            pp = np.array(c_path.xy, dtype=np.int32)
            ax.plot(pp[0, 0], pp[0, 1], "o", color='gray', markersize=3)
            ax.plot(pp[-1, 0], pp[-1, 1], "o", color='gray', markersize=3)
            ax.plot(pp[:, 0], pp[:, 1], color='gray', linestyle=':')

    def save_data(self):
        len_px = self.dec_interface.decomposed.smcells_seq_length_px
        len_m = len_px * self.resolution
        data = {"centers_seq_length": round(len_m, 2)}
        with open(self.zmd.result_file, 'w') as f:
            json.dump(data, f, indent=4)
            f.write("\n")

    def plot_paths_m(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("paths_m")
        #
        n = self.dec_interface.decomposed.n_mcells
        colors = get_n_colors_plt(n)
        for i, cell_covp_planner in enumerate(self.all_cell_covp_planners):
            path = cell_covp_planner.path_m.xy
            if len(path) < 1:
                continue
            pp = np.array(path, dtype=np.float32)
            ax.plot(pp[:, 0], pp[:, 1], color=colors[i], linewidth=2, linestyle='-')
        #
        for c_path in self.c_paths_m:
            if len(c_path) < 2:
                continue
            pp = np.array(c_path, dtype=np.float32)
            ax.plot(pp[:, 0], pp[:, 1], color='gray', linestyle=':')
        plt.show()
