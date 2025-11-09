""" Create district for coverage path planning. Simple version: select whole map as district. """
from covpp.gui.rectangle import Rectangle
from covpp.base.logger import MyLogger
from covpp.base.map_data import MapData
from covpp.base.configs import PipeLineConfig


class DistrictGen:
    """ Create district for coverage path planning. Simple version: select whole map as district. """

    def __init__(self, map_data: MapData):
        self.lg = MyLogger("DistrictEvent")
        self.lg.info1("Step: creating district for CovPP based on simple method (whole map)")

        # attributes
        self.__md: MapData = map_data
        self.__cfg = PipeLineConfig()

        # Check if whole map is selected
        if not self.__cfg.whole_map:
            raise ValueError("DistrictGen is not implemented for non-whole map configurations.")

    def get_district_rect(self) -> Rectangle:
        """ Get district rectangle based on map data."""
        rot = 0
        xc = self.__md.up_c_w
        yc = self.__md.up_c_h
        w = self.__md.updated_w
        h = self.__md.updated_h
        data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        return Rectangle(data)
