# from scripts.base.configs import SpaceConfigs  # BConfigs
from scripts.district.district import DistrictData, ZoneData


class CPPSimple:
    district: DistrictData  # main district data

    def __init__(self, district: DistrictData):
        self.ns = "CPPSimple"
        print(f"[{self.ns}]: Step: coverage path planning")

        #
        self.district = district
        self.cpp_simple()

    def cpp_simple(self):
        print(f"[{self.ns}]: coverage path planning")

        # for each zone # list[ZoneData]
        for zd in self.district.zone_datas:
            self.zone_cpp(zd)

    def zone_cpp(self, zd: ZoneData):
        print(f"[{self.ns}]: cpp for zone [{zd.zone_map_data.ind}]")
