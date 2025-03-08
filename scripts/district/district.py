from copy import deepcopy
import numpy as np
from scripts.gui.visuals import Rectangle
from scripts.district.zone import ZoneMapData
from scripts.base.space_analyzer import SpaceAnalyzer
# from scripts.base.my_utils import visualize_map_s


class ZoneData:
    zone_map_data: ZoneMapData

    def __init__(self):
        self.zone_map_data = None


class DistrictData:
    dmd: ZoneMapData  # district_map_data

    def __init__(self, spa: SpaceAnalyzer):
        self.ns = "DistrictData"

        # main district map data
        print(f"[{self.ns}]: creating district data")
        self.dmd = spa.zone_map_data
        self.dmd.visualize()

        # save district data
        base_path = self.dmd.orig_map_data.bcfg.district_folder
        district_path = base_path + "/District"
        self.dmd.save_data(district_path)

        # # zones
        zone_datas: list[ZoneData] = []
        # create zones rectangles
        zone_rectangles: list[Rectangle] = self.create_zone_rectangles_2()
        # create zone ws maps
        for i, zr in enumerate(zone_rectangles):
            # create zone map data
            print(f"[{self.ns}]: creating map data for zone [{i}]")
            z_spa = deepcopy(spa)
            z_spa.update_by_rectangle(zr)
            zmd = z_spa.zone_map_data  # zone_map_data
            zmd.set_ind(i)

            # save zone data
            zone_path = base_path + f"/Zone_{i}"
            zmd.save_data(zone_path)

            # append zone map data
            zd = ZoneData()
            zd.zone_map_data = zmd
            zone_datas.append(zd)

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

    # TODO:
    # get zones rectangles
    # get zones map data
    # apply wf path on zones
    # apply cpp on zones
