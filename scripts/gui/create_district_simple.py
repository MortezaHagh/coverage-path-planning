from scripts.base.map_data import MapData
from scripts.base.configs import DistrictConfgs
from scripts.gui.visuals import Rectangle


class DistrictGen:
    __ns: str
    __md: MapData
    __cfg: DistrictConfgs

    def __init__(self, map_data: MapData):
        #
        self.__ns = "DistrictEvent"
        print(f" >>>>> [{self.__ns}] Step: Creating district for CPP")

        #
        self.__md = map_data
        self.__cfg = DistrictConfgs()

        #
        if self.__cfg.whole_map:
            print(f"[{self.__ns}]: whole map is selected as district for CPP!")
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
