import cv2 as cv
import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.planning.contour.cell_cov_planner import CellCoveragePathPlanner
from scripts.planning.contour.decomposition_interface import DecompositionInterface
# from scripts.planning.connect_paths import ConnectPaths
# from scripts.planning.cells_graph_1 import CellsGraph


class CoveragePP(CoveragePPBase):
    """  contour based coverage path planning."""

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns: str = "ContourCoveragePP" + "-Z" + str(self.zmd.ind)
        self.lg.name = self.ns
        self.lg.info1("Step: coverage path planning")

        # main
        self.dec_interface: DecompositionInterface = None                   # voronoi decomposition interface
        self.all_cell_covp_planners: list[CellCoveragePathPlanner] = []

    def decompose_map(self) -> None:
        self.lg.info2("start voronoi decomposition")
        self.dec_interface = DecompositionInterface(self.zmd, self.map_polygons)

    def zone_covpp(self) -> None:
        self.lg.info2("start coverage path planning for cells")
        for mcell in self.dec_interface.mcells:
            cell_covp_planner = CellCoveragePathPlanner(cell=mcell, zmd=self.zmd, zone_ind=self.zmd.ind)
            self.all_cell_covp_planners.append(cell_covp_planner)
            if cell_covp_planner.path_m:
                mcell.set_path_m(cell_covp_planner.path_m)
                mcell.set_path_px(cell_covp_planner.path_px)
            else:
                mcell.do_planning = False
                self.lg.warn(f"Cell {mcell.id} has no path after coverage planning.")

    # plot and save ====================================================================================================

    def add_cell_paths_to_map_planning(self) -> None:
        map_planning = self.zmd.map_planning.copy()
        robot_r_px = int(self.pcfgs.cleaning_r_ov / self.resolution)
        robot_d_px = robot_r_px * 2
        p_r = 5  # int(robot_r_px/10)
        for i, path_px in enumerate(self.paths_px):
            if path_px:
                pp = np.array(path_px.xy, dtype=np.int32)  # Ensure pp is a numpy array of integers
                map_planning = cv.polylines(map_planning, [pp], isClosed=False, color=self.colors[i], thickness=2)
                # plot cleaning
                overlay = map_planning.copy()
                cv.polylines(overlay, [pp], isClosed=False, color=self.colors[i], thickness=robot_d_px)
                alpha = 0.2  # Adjust transparency (0.0 transparent, 1.0 opaque)
                map_planning = cv.addWeighted(overlay, alpha, map_planning, 1 - alpha, 0)
                # add points
                for point in pp:
                    cv.circle(map_planning, tuple(point), p_r, self.colors[i], -1)
        self.zmd.map_planning = map_planning
