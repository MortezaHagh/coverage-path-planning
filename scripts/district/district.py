from copy import deepcopy
import numpy as np
from scripts.gui.visuals import Rectangle
from scripts.district.zone import ZoneMapData
from scripts.base.space_analyzer import SpaceAnalyzer
# from scripts.base.my_utils import visualize_map_s


class ZoneData:
    zone_map_data: ZoneMapData
    path: list[tuple[int, int]]

    def __init__(self):
        self.zone_map_data = None
        self.path = None


class DistrictData:
    spa_obj: SpaceAnalyzer      # space_analyzer
    dmd: ZoneMapData            # district_map_data
    zone_datas: list[ZoneData]  # zones

    def __init__(self, spa: SpaceAnalyzer):
        self.ns = "DistrictData"
        print(f"[{self.ns}]: Step: creating district data")

        # main district map data
        self.spa_obj = spa
        self.dmd = spa.zone_map_data
        self.dmd.visualize()

        # save district data
        base_path = self.dmd.orig_map_data.bcfg.district_folder
        district_path = base_path + "/District"
        self.dmd.save_data(district_path)

        # create zones
        self.zone_datas: list[ZoneData] = []
        # self.create_zones()

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
            base_path = self.dmd.orig_map_data.bcfg.district_folder
            zone_path = base_path + f"/Zone_{i}"
            zmd.save_data(zone_path)

            # append zone map data
            zd = ZoneData()
            zd.zone_map_data = zmd
            self.zone_datas.append(zd)
        print(f"[{self.ns}]: ----------------------------------------------- ")

    def create_zone_rectangles_1(self) -> list[Rectangle]:
        rect = self.dmd.rectangle
        dx1 = -np.sin(rect.rot) * (rect.height//4)
        dy1 = np.cos(rect.rot) * (rect.height//4)
        xc1 = int(rect.xc + dx1)
        yc1 = int(rect.yc + dy1)
        dx2 = np.sin(rect.rot) * (rect.height//4)
        dy2 = -np.cos(rect.rot) * (rect.height//4)
        xc2 = int(rect.xc + dx2)
        yc2 = int(rect.yc + dy2)
        data1 = {"xc": xc1, "yc": yc1, "rot": rect.rot, "w": rect.width, "h": rect.height//2}
        data2 = {"xc": xc2, "yc": yc2, "rot": rect.rot, "w": rect.width, "h": rect.height//2}
        rect1 = Rectangle(data1)
        rect2 = Rectangle(data2)
        return [rect1, rect2]

    def create_zone_rectangles_2(self) -> list[Rectangle]:
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
