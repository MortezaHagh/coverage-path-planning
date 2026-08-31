"""District data management module for zone creation and coverage planning orchestration."""
from copy import deepcopy
import cv2 as cv
import numpy as np
from covpp.base.logger import MyLogger
from covpp.gui.rectangle import Rectangle
from covpp.district.zone import ZoneData
from covpp.base.configs import PipeLineConfig
from covpp.spa.space_analyzer import SpaceAnalyzer
from covpp.district.zone_map_data import ZoneMapData
from covpp.evaluation.district_evaluation import DistrictEvaluationHandler, DistrictStatsBase


class DistrictData:
    """Manage district-level data, zones, and coordinate coverage planning operations.

    This class serves as the high-level orchestrator for the coverage planning
    pipeline at the district level. It creates and manages zones, coordinates
    coverage planning across all zones, and aggregates results.

    Args:
        spa (SpaceAnalyzer): SpaceAnalyzer object containing processed workspace data.

    Attributes:
        spa_obj (SpaceAnalyzer): Reference to the SpaceAnalyzer object.
        dmd (ZoneMapData): District map data.
        zone_datas (list[ZoneData]): List of ZoneData objects representing subdivisions.
        district_evaluator (DistrictEvaluationHandler): Handler for district-level statistics evaluation.
        district_stats (DistrictStatsBase): Aggregated district statistics object.

    Example:
        >>> district = DistrictData(space_analyzer)
        >>> district.create_zones_hl()
        >>> district.run_planning_pipeline()
    """

    def __init__(self, spa: SpaceAnalyzer):
        self.lg = MyLogger("DistrictData")
        self.lg.info1("Step: creating district data")

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
        self.district_evaluator = DistrictEvaluationHandler(self.dmd.result_file)
        self.district_stats = DistrictStatsBase()

    def create_zones_hl(self) -> None:
        """Create zones and initialize district-level planning visualization.

        Steps:
        - Generate zone rectangles and `ZoneMapData`
        - Construct `ZoneData` objects for each zone
        - Update the district planning map overlay
        """
        self.create_zones()
        self.update_map_planning()

    def run_planning_pipeline(self) -> None:
        """Run segmentation → coverage → sequencing → evaluation for all zones.

        Result:
        - Updates `district_stats`, the final planning map, and persists outputs.
        """

        self.lg.info1("Step: running coverage planning pipeline on all zones")

        zone_stats = []
        for zd in self.zone_datas:
            # self.lg.info(" ------------------------------------------------ ")
            # self.lg.info(f" ---------- Planning on Zone: {zd.ind} ------------- ")
            # self.lg.info(" ------------------------------------------------ ")
            zd.zone_planning_pipeline()  # run full planning pipeline on zone
            zone_stats.append(zd.zone_stats)  # collect zone stats

        # temp
        # stats
        self.lg.info1("Step: evaluating district statistics from all zones")
        self.district_evaluator.evaluate(zone_stats)
        self.district_stats = self.district_evaluator.stats

        # final map planning
        self.update_map_planning()
        self.save_map_planning()

    def create_zones(self) -> None:
        """Create zones using the configured rectangle subdivision method.

        Steps:
        - Create zone rectangles (currently single rectangle per district)
        - Clone SpaceAnalyzer, update by zone rectangle, and fetch `ZoneMapData`
        - Instantiate `ZoneData` and register in `zone_datas`
        """
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
        """Composite district planning map from per-zone planning overlays."""
        mapp = self.dmd.map_planning.copy()
        for zd in self.zone_datas:
            x1, y1 = zd.zmd.hw_bd["h_min"], zd.zmd.hw_bd["w_min"]
            x2, y2 = zd.zmd.hw_bd["h_max"], zd.zmd.hw_bd["w_max"]
            inds = zd.zmd.map_sf_orig > 0
            # copy zone map planning to district map planning
            mapp[x1:x2, y1:y2][inds] = zd.zmd.map_planning[x1:x2, y1:y2][inds]
        self.dmd.map_planning = mapp

    def save_map_planning(self) -> None:
        """Save district planning visualization as a lossless PNG (no compression). """
        if not PipeLineConfig.do_save_maps:
            return
        m_name = f"/mp_{PipeLineConfig.method}.png"
        map_planning_path = self.dmd.zone_dir + m_name
        cv.imwrite(map_planning_path, self.dmd.map_planning, [cv.IMWRITE_PNG_COMPRESSION, 0])

    # ==================================================================================================================
    # Zone Rectangle Creation Methods
    # ==================================================================================================================

    def create_zone_rectangles_1(self) -> list[Rectangle]:
        """Split the district into two halves along the short axis (rotated). """
        # data = self.dmd.rect_data
        data = self.dmd.final_rect_data
        h = data["h"]
        w = data["w"]
        rot = data["rot"]
        dx1 = -np.sin(rot) * (h // 4)
        dy1 = np.cos(rot) * (h // 4)
        xc1 = int(data["xc"] + dx1)
        yc1 = int(data["yc"] + dy1)
        dx2 = np.sin(rot) * (h // 4)
        dy2 = -np.cos(rot) * (h // 4)
        xc2 = int(data["xc"] + dx2)
        yc2 = int(data["yc"] + dy2)
        data1 = {"xc": xc1, "yc": yc1, "rot": rot, "w": w, "h": h // 2}
        data2 = {"xc": xc2, "yc": yc2, "rot": rot, "w": w, "h": h // 2}
        rect1 = Rectangle(data1)
        rect2 = Rectangle(data2)
        return [rect1, rect2]

    def create_one_zone_rectangle(self) -> list[Rectangle]:
        """Create a single zone rectangle covering the whole district. """
        data = self.dmd.final_rect_data
        h = data["h"]
        w = data["w"]
        rot = data["rot"]
        xc = int(data["xc"])
        yc = int(data["yc"])
        data1 = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        rect1 = Rectangle(data1)
        return [rect1]
