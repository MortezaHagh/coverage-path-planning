# from scripts.base.configs import SpaceConfigs  # BConfigs
from scripts.district.district import DistrictData


class CPPSimple:
    district: DistrictData  # main district data

    def __init__(self, district: DistrictData):
        self.ns = "CPPSimple"
        print(f"[{self.ns}]: Step: coverage path planning")

        #
        self.district = district

    def cpp_simple(self):
        print(f"[{self.ns}]: coverage path planning")
        # for each zone
        for i, zd in enumerate(self.district.zone_datas):
            print(f"[{self.ns}]: zone [{i}]")
            # path planning
            self.zone_cpp(zd)
