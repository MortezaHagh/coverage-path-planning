import os
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase
from scripts.planning.decomposition_interface import DecompositionInterface


class CoveragePP(CoveragePPBase):
    """  DecCoveragePP  """

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "DecCoveragePP" + "-" + str(self.zmd.ind)
        print(f"[{self.ns}]: Step: coverage path planning")

        #
        self.dec: DecompositionInterface = None

    def coverage_planning(self):
        print(f"[{self.ns}]: BS coverage path planning")
        self.trap_dec_cpp()
        # self.save_decomposition()

    # =================================================================================================================

    def trap_dec_cpp(self):
        print(f"[{self.ns}]: start decomposition")
        self.dec = DecompositionInterface(self.zmd)

    def save_decomposition(self):
        base_path = self.zmd.orig_map_data.bcfg.district_dir
        dec_path = base_path + self.zmd.name
        dec_path = os.path.join(dec_path, "Dec1")
        self.dec.decomposed.save_figs(dec_path)

    # =================================================================================================================

    def coverage_planning_l(self):
        print(f"[{self.ns}]: BS coverage path planning")
        self.zone_covpp()
        self.rotate_waypoints()
        self.create_path_px()

    def zone_covpp(self):
        pass
