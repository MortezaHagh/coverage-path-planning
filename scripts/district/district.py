from copy import deepcopy
import cv2 as cv
import numpy as np
from scripts.gui.visuals import Rectangle
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.space_analyzer import SpaceAnalyzer
from scripts.district.zone import ZoneData


class DistrictData:
    spa_obj: SpaceAnalyzer      # space_analyzer
    dmd: ZoneMapData            # district_map_data
    zone_datas: list[ZoneData]  # zones
    #
    map_hl: np.ndarray          # map high level

    def __init__(self, spa: SpaceAnalyzer):
        self.ns = "DistrictData"
        print(f"[{self.ns}]: Step: creating district data")

        # main district map data
        self.spa_obj = spa
        self.dmd = spa.zone_map_data
        # self.dmd.visualize()

        # save district data
        self.save_init_district_data()

        # attributes
        self.map_hl = None  # map info: containing zones rect, paths, ...
        self.zone_datas: list[ZoneData] = []  # zones

    def create_zones_hl(self):
        """create all zones in the district and create map_hl (high level)"""
        self.create_zones()
        self.create_map_hl()
        self.save_map_hl()

    def create_zones(self):
        print(f"[{self.ns}]: Step: creating zone map data")
        # create zones rectangles
        zone_rectangles: list[Rectangle] = self.create_zone_rectangles_2()
        # create zone ws maps
        for i, zr in enumerate(zone_rectangles):
            # create zone map data
            print(f"[{self.ns}]: creating map data for zone [{i}]")
            z_spa: SpaceAnalyzer = deepcopy(self.spa_obj)
            z_spa.update_by_rectangle(zr)
            zmd = z_spa.zone_map_data  # zone_map_data
            zmd.set_ind(i)

            # save zone data
            zone_path = f"Zone_{i}"
            zmd.save_data(zone_path)

            # append zone map data
            zd = ZoneData(i)
            zd.zone_map_data = zmd
            self.zone_datas.append(zd)
        print(f"[{self.ns}]: ----------------------------------------------- ")

    def save_init_district_data(self):
        district_path = "District"
        self.dmd.save_data(district_path)

    def create_map_hl(self):
        map_hl = self.dmd.map_viz.copy()
        self.map_hl = map_hl
        # display zones rectangles
        for zd in self.zone_datas:
            vertices = zd.zone_map_data.vertices
            self.map_hl = cv.polylines(self.map_hl, [vertices], isClosed=True, color=(100, 0, 150), thickness=4)
            # for zone
            zd.create_map_hl()

    def save_map_hl(self):
        base_path = self.dmd.orig_map_data.bcfg.district_dir
        map_hl_path = base_path + "/District"
        cv.imwrite(map_hl_path + "/" + "map_hl.png", self.map_hl)

    def add_os_path_map_hl(self):
        """add os path of all zones to map_hl"""
        for zd in self.zone_datas:
            path = zd.os_path_m
            for (p1, p2) in zip(path[:-1], path[1:]):
                p1 = self.point_m_to_px(p1)
                p2 = self.point_m_to_px(p2)
                self.map_hl = cv.line(self.map_hl, p1, p2, (0, 255, 10), 4)
                # for zone
                zd.add_os_path_map_hl()
        self.save_map_hl()

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        res = self.dmd.reslolution
        return (int(p[0] / res), int(p[1] / res))

    def create_zone_rectangles_1(self) -> list[Rectangle]:
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
