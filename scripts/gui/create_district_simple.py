from scripts.gui.visuals import Rectangle
from scripts.base.logger import MyLogger
from scripts.base.map_data import MapData
from scripts.base.configs import DistrictConfigs


class DistrictGen:

    def __init__(self, map_data: MapData):
        #
        self.__ns: str = "DistrictEvent"
        self.lg = MyLogger(self.__ns)
        self.lg.info1("Step: creating district for CovPP")

        #
        self.__md: MapData = map_data
        self.__cfg: DistrictConfigs = DistrictConfigs()

        #
        if self.__cfg.whole_map:
            self.lg.info2("whole map is selected as district for CPP!")
        else:
            raise ValueError("DistrictGen is not implemented for non-whole map configurations.")

    def get_district_rect(self) -> Rectangle:
        # create district based on map data
        rot = 0
        xc = self.__md.up_c_w
        yc = self.__md.up_c_h
        w = self.__md.updated_w
        h = self.__md.updated_h
        data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        return Rectangle(data)
