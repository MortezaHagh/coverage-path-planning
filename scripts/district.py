from copy import deepcopy
from scripts.zone import ZoneMapData
from scripts.gui.visuals import Rectangle
from scripts.base.space_analyzer import SpaceAnalyzer
# from scripts.base.my_utils import visualize_map_s


class DistrictData:
    dmd: ZoneMapData  # district_map_data

    def __init__(self, spa: SpaceAnalyzer):
        self.ns = "DistrictData"

        # main district map data
        print(f"[{self.ns}]: creating district data")
        self.dmd = spa.zone_map_data
        self.dmd.visualize()

        # save district data
        base_path = self.dmd.map_data.bcfg.district_folder
        district_path = base_path + "/District"
        self.dmd.save_data(district_path)

        # # zones
        zone_map_datas: list[ZoneMapData] = []
        # create zones rectangles
        zone_rectangles: list[Rectangle] = self.create_zone_rectangles()
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
            zone_map_datas.append(zmd)

    def create_zone_rectangles(self) -> list[Rectangle]:
        height, width = self.dmd.hw
        x1 = width//2
        y1 = height//4
        x2 = width//2
        y2 = 3*height//4
        w1 = width
        h1 = height//2
        w2 = width
        h2 = height//2
        data1 = {"xc": x1, "yc": y1, "rot": 0, "w": w1, "h": h1}
        data2 = {"xc": x2, "yc": y2, "rot": 0, "w": w2, "h": h2}
        rect1 = Rectangle(data1)
        rect2 = Rectangle(data2)
        return [rect1, rect2]

    # TODO:
    # get zones rectangles
    # get zones map data
    # apply wf path on zones
    # apply cpp on zones
