""" District data class. Includes all zones data and district map data. """
from copy import deepcopy
import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.gui.rectangle import Rectangle
from scripts.district.zone import ZoneData
from scripts.base.configs import PipeLineConfig
from scripts.spa.space_analyzer import SpaceAnalyzer
from scripts.district.zone_map_data import ZoneMapData
from scripts.evaluation.district_evaluation import DistrictEvaluationHandler, DistrictStatsBase


class DistrictData:
    """
    District data class. Includes all zones data and district map data.

    Methods:
        create_zones_hl: create all zones in the district and create map_planning (high level).
        run_coverage_planning: run coverage planning for all zones.

    Attributes:
        spa_obj: SpaceAnalyzer - space analyzer object.
        dmd: ZoneMapData - main district map data.
        zone_datas: list[ZoneData] - list of zones in the district.
        evaluator: DistrictEvaluationHandler - evaluator for district statistics.
        stats: DistrictStatsBase - statistics for the district.
    """

    def __init__(self, spa: SpaceAnalyzer):
        self.lg = MyLogger("DistrictData")
        self.lg.info1("Step: creating district data")

        # configs
        self.dcfg: PipeLineConfig = PipeLineConfig()

        # space_analyzer object
        self.spa_obj: SpaceAnalyzer = spa

        # main district map data
        self.dmd: ZoneMapData = spa.zone_map_data
        self.dmd.set_name("District")

        # save district map data to files
        self.dmd.save_data()

        # attributes
        self.zone_datas: list[ZoneData] = []  # zones

        # statistics
        self.evaluator = DistrictEvaluationHandler(self.dmd.result_file)
        self.stats = DistrictStatsBase()

    def create_zones_hl(self) -> None:
        """create all zones in the district and create map_planning (high level). """
        self.create_zones()
        self.update_map_planning()
        # self.save_map_planning()

    def run_coverage_planning(self) -> None:
        """run coverage planning for all zones. Then, evaluate district stats."""

        zone_stats = []
        for zd in self.zone_datas:
            self.lg.info1(f" === Planning on Zone: {zd.ind} ================== ")
            zd.zone_planning_pipeline()
            zone_stats.append(zd.stats)

        # stats
        self.evaluator.evaluate(zone_stats)
        self.stats = self.evaluator.stats

        #
        self.update_map_planning()
        self.save_map_planning()

    def create_zones(self) -> None:
        """ create zones based on the rectangle method. """
        self.lg.info1("Step: creating zones [ZoneData] and their map data")

        # create zones rectangles
        zone_rectangles: list[Rectangle] = self.create_one_zone_rectangle()

        # create all zones [zone data(zone_map_data)]
        for i, zr in enumerate(zone_rectangles):
            self.lg.info(f"creating map data and creating zone [{i}]")
            z_spa: SpaceAnalyzer = deepcopy(self.spa_obj)
            z_spa.update_by_rectangle(zr)   # space analyzer updated by zone rectangle
            zmd = z_spa.zone_map_data       # zone_map_data from space analyzer
            zmd.set_ind(i)
            # create zone
            zd = ZoneData(i, zmd)
            self.zone_datas.append(zd)

    def update_map_planning(self) -> None:
        """ update district map_planning based on all zones map_planning. """
        mapp = self.dmd.map_planning.copy()
        for zd in self.zone_datas:
            x1, y1 = zd.zmd.hw_bd_px["h_min"], zd.zmd.hw_bd_px["w_min"]
            x2, y2 = zd.zmd.hw_bd_px["h_max"], zd.zmd.hw_bd_px["w_max"]
            inds = zd.zmd.map_sf_orig > 0
            # copy zone map planning to district map planning
            mapp[x1:x2, y1:y2][inds] = zd.zmd.map_planning[x1:x2, y1:y2][inds]
        self.dmd.map_planning = mapp

    def save_map_planning(self) -> None:
        """ save district map_planning to file. """
        if not self.dcfg.save_map_planning:
            return
        m_name = f"/map_planning_{self.dcfg.method}.png"
        map_planning_path = self.dmd.zone_dir + m_name
        cv.imwrite(map_planning_path, self.dmd.map_planning)

    # ==================================================================================================================

    def create_zone_rectangles_1(self) -> list[Rectangle]:
        """ create two zone rectangles by splitting the district rectangle into two halves. """
        data = self.dmd.rect_data
        h = data["h"]
        w = data["w"]
        rot = data["rot"]
        dx1 = -np.sin(rot) * (h//4)
        dy1 = np.cos(rot) * (h//4)
        xc1 = int(data["xc"] + dx1)
        yc1 = int(data["yc"] + dy1)
        dx2 = np.sin(rot) * (h//4)
        dy2 = -np.cos(rot) * (h//4)
        xc2 = int(data["xc"] + dx2)
        yc2 = int(data["yc"] + dy2)
        data1 = {"xc": xc1, "yc": yc1, "rot": rot, "w": w, "h": h//2}
        data2 = {"xc": xc2, "yc": yc2, "rot": rot, "w": w, "h": h//2}
        rect1 = Rectangle(data1)
        rect2 = Rectangle(data2)
        return [rect1, rect2]

    def create_zone_rectangles_2(self) -> list[Rectangle]:
        """ create two zone rectangles by splitting the final district rectangle into two halves. """
        data = self.dmd.final_rect_data
        h = data["h"]
        w = data["w"]
        rot = data["rot"]
        dx1 = -np.sin(rot) * (h//4)
        dy1 = np.cos(rot) * (h//4)
        xc1 = int(data["xc"] + dx1)
        yc1 = int(data["yc"] + dy1)
        dx2 = np.sin(rot) * (h//4)
        dy2 = -np.cos(rot) * (h//4)
        xc2 = int(data["xc"] + dx2)
        yc2 = int(data["yc"] + dy2)
        data1 = {"xc": xc1, "yc": yc1, "rot": rot, "w": w, "h": h//2}
        data2 = {"xc": xc2, "yc": yc2, "rot": rot, "w": w, "h": h//2}
        rect1 = Rectangle(data1)
        rect2 = Rectangle(data2)
        return [rect1, rect2]

    def create_one_zone_rectangle(self) -> Rectangle:
        """ create one zone rectangle covering the whole district. """
        data = self.dmd.final_rect_data
        h = data["h"]
        w = data["w"]
        rot = data["rot"]
        xc = int(data["xc"])
        yc = int(data["yc"])
        data1 = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        rect1 = Rectangle(data1)
        return [rect1]
