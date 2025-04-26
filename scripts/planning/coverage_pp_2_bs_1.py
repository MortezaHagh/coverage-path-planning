import os
import cv2 as cv
import numpy as np
from scripts.base.configs import SpaceConfigs
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.planning.decomposition_interface import DecompositionInterface
from scripts.planning.cell_bs_cov_planner import CellCoveragePP


class CoveragePP(CoveragePPBase):
    """  DecCoveragePP  """

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "DecCoveragePP" + "-" + str(self.zmd.ind)
        print(f"[{self.ns}]: Step: coverage path planning")

        # all_ccp
        self.all_ccpp: list[CellCoveragePP] = []
        self.cell_paths_px: list[tuple[int, int]] = []

        # run decomposition and coverage planning
        self.coverage_planning()

    def coverage_planning(self):
        print(f"[{self.ns}]: BS coverage path planning")
        self.decompose_map()
        # self.save_decomposition()
        self.add_cells_to_map_planning()
        #
        self.coverage_planning_l()

    # =================================================================================================================

    def decompose_map(self):
        print(f"[{self.ns}]: start decomposition")
        self.dec_interface = DecompositionInterface(self.zmd)

    def save_decomposition(self):
        base_path = self.zmd.orig_map_data.bcfg.district_dir
        dec_path = base_path + self.zmd.name
        dec_path = os.path.join(dec_path, "Dec1")
        self.dec_interface.decomposed.save_figs(dec_path)

    def add_cells_to_map_planning(self):
        for cell in self.dec_interface.decomposed.cells_groups:
            pp = cell.merged_cell.poly_points
            pp = np.array(pp, dtype=np.int32)  # Ensure pp is a numpy array of integers
            self.zmd.map_planning = cv.polylines(self.zmd.map_planning, [pp], isClosed=True, color=(0, 0, 0), thickness=6)

    # =================================================================================================================

    def coverage_planning_l(self):
        print(f"[{self.ns}]: BS coverage path planning")
        self.zone_covpp()
        self.add_cell_paths_to_map_planning()
        # self.create_final_path()

    def zone_covpp(self):
        for cellg in self.dec_interface.decomposed.cells_groups:
            ccpp = CellCoveragePP(cell=cellg.merged_cell, resolution=self.resolution)
            self.all_ccpp.append(ccpp)
            self.cell_paths_px.extend(ccpp.path_px)

    def create_final_path(self):
        self.path_px = self.cell_paths_px.copy()

    def add_cell_paths_to_map_planning(self):
        # pp = np.array(self.cell_paths_px, dtype=np.int32)  # Ensure pp is a numpy array of integers
        # self.zmd.map_planning = cv.polylines(self.zmd.map_planning, [pp], isClosed=False, color=(0, 255, 0), thickness=2)
        #
        scfg = SpaceConfigs()
        for ccpp in self.all_ccpp:
            path_px = ccpp.path_px
            if path_px:
                pp = np.array(path_px, dtype=np.int32)  # Ensure pp is a numpy array of integers
                self.zmd.map_planning = cv.polylines(self.zmd.map_planning, [pp], isClosed=False, color=(0, 255, 0), thickness=2)
                #
                radius_px_3 = int(np.ceil(scfg.cleaning_r / self.zmd.orig_map_data.up_resolution))
                pose = self.point_m_to_px(pp[0])
                cv.circle(self.zmd.map_planning, pose, radius_px_3, [100, 0, 100], -1)
