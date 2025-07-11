import cv2 as cv
import numpy as np
from scripts.base.my_utils import get_n_colors_cv
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.connect_paths import ConnectPaths
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.planning.contour.cell_cov_planner import CellCoveragePathPlanner
from scripts.planning.contour.decomposition_interface import DecompositionInterface


class CoveragePP(CoveragePPBase):
    """  contour based coverage path planning."""
    colors: list
    dec_interface: DecompositionInterface
    all_cell_covp_planners: list[CellCoveragePathPlanner]
    connections: ConnectPaths

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "ContourCoveragePP" + "-Z" + str(self.zmd.ind)
        self.lg.name = self.ns
        self.lg.info1("Step: coverage path planning")

        # configs
        self.colors = []

        # main
        self.connections = ConnectPaths(self.zmd)   # connect paths interface
        self.dec_interface = None                   # voronoi decomposition interface
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

        # run decomposition and coverage planning
        self.plan()

    def coverage_planning(self) -> None:
        self.decompose_map()
        self.zone_covpp()
        self.create_sequence()
        self.connect_paths()

    def decompose_map(self) -> None:
        self.lg.info2("start voronoi decomposition")
        self.dec_interface = DecompositionInterface(self.zmd, self.map_polygons)

    def zone_covpp(self) -> None:
        self.lg.info2("start coverage path planning for cells")
        for cell in self.dec_interface.mcells:
            cell_covp_planner = CellCoveragePathPlanner(cell=cell, zmd=self.zmd, zone_ind=self.zmd.ind)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                cell.set_path_m(cell_covp_planner.path_m)
                cell.set_path_px(cell_covp_planner.path_px)
            else:
                cell.do_planning = False
                self.lg.warn(f"Cell {cell.id} has no path after coverage planning.")

    def create_sequence(self) -> None:
        if self.dcfgs.cell_sequence_method == "s3_milp":
            # calculate paths and distances between cells
            self.connections.find_all_connections(self.dec_interface.mcells)
            self.connections.find_all_distances()
        #
        self.dec_interface.create_sequence(self.connections.all_distances)
        self.paths_m = [cell.path_m for cell in self.dec_interface.smcells if cell.do_planning]
        self.paths_px = [cell.path_px for cell in self.dec_interface.smcells if cell.do_planning]

    def connect_paths(self) -> None:
        self.connections.connect_paths(self.paths_px, self.dec_interface.seq_idxs)
        self.c_paths_px = self.connections.c_paths_px
        self.c_paths_m = [self.path_px_to_m(path_px) for path_px in self.c_paths_px]

    def method_post_planning(self) -> None:
        if not self.dcfgs.save_map_planning:
            return
        self.plotter.run(self.dec_interface.mcells)
        n = self.dec_interface.n_mcells
        self.colors = get_n_colors_cv(n)
        self.add_cells_to_map_planning()
        self.add_cell_paths_to_map_planning()

    # plot and save ====================================================================================================

    def add_cells_to_map_planning(self) -> None:
        for i, cell in enumerate(self.dec_interface.mcells):
            pp = cell.points
            pp = np.array(pp, dtype=np.int32)  # Ensure pp is a numpy array of integers
            map_p = cv.polylines(self.zmd.map_planning, [pp], isClosed=True, color=(100, 100, 100), thickness=2)
            map_p_fill = self.zmd.map_planning.copy()
            cv.fillPoly(map_p_fill, [pp], color=self.colors[i])
            alpha = 0.1
            self.zmd.map_planning = cv.addWeighted(map_p_fill, alpha, map_p, 1 - alpha, 0)

    def add_cell_paths_to_map_planning(self) -> None:
        map_planning = self.zmd.map_planning.copy()
        robot_r_px = int(self.pcfgs.cleaning_r_ov / self.resolution)
        robot_d_px = robot_r_px * 2
        p_r = 5  # int(robot_r_px/10)
        for i, path_px in enumerate(self.paths_px):
            if path_px:
                pp = np.array(path_px.xy, dtype=np.int32)  # Ensure pp is a numpy array of integers
                map_planning = cv.polylines(map_planning, [pp], isClosed=False, color=self.colors[i], thickness=2)
                overlay = map_planning.copy()
                cv.polylines(overlay, [pp], isClosed=False, color=self.colors[i], thickness=robot_d_px)
                alpha = 0.2  # Adjust transparency (0.0 transparent, 1.0 opaque)
                map_planning = cv.addWeighted(overlay, alpha, map_planning, 1 - alpha, 0)
                # add points
                for point in pp:
                    cv.circle(map_planning, tuple(point), p_r, self.colors[i], -1)
        self.zmd.map_planning = map_planning
