import cv2 as cv
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.coverage_pp_2_bs_1 import CoveragePP
from scripts.base.configs import SpaceConfigs, DistrictConfgs
# from scripts.planning.coverage_pp_1_simple import CoveragePP


class ZoneData:
    ind: int
    zone_map_data: ZoneMapData
    cov_planner: CoveragePP
    cleaning_area: float

    def __init__(self, ind: int = None):
        self.ind = ind
        self.zone_map_data = None

        # configs
        self.scfg = SpaceConfigs()
        self.dcfg = DistrictConfgs()

        # coverage path planner
        self.cov_planner = None

        # statistics - todo
        self.cleaning_area = None

    def set_map_data(self, zone_map_data: ZoneMapData):
        self.zone_map_data = zone_map_data
        self.initialize()

    def initialize(self):
        self.zone_map_data.save_data()
        self.create_map_planning()
        # self.save_map_planning()

    def coverage_planning(self):
        self.cov_planner = CoveragePP(self.zone_map_data)
        self.save_map_planning()

    # ==================================================================================================================

    def create_map_planning(self):
        map_planning = self.zone_map_data.map_viz.copy()
        vertices = self.zone_map_data.vertices
        map_planning = cv.polylines(map_planning, [vertices], isClosed=True, color=(100, 0, 150), thickness=2)
        self.zone_map_data.map_planning = map_planning

    def save_map_planning(self):
        if not self.dcfg.save_map_planning:
            return
        base_path = self.zone_map_data.orig_map_data.bcfg.district_dir
        map_hl_path = base_path + f"/Zone_{self.ind}"
        cv.imwrite(map_hl_path + "/" + "map_planning.png", self.zone_map_data.map_planning)

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        res = self.zone_map_data.reslolution
        return (int(p[0] / res), int(p[1] / res))
