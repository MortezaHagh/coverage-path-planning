from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfigs
from scripts.segmentation.cell_base import CellBase
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData


class SegmentationBase:
    """ Base class for Segmentation.  
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        self.lg = MyLogger("SegmentationBase")

        # zone map data
        self.zmd: ZoneMapData = zmd

        # configs
        self.dcfgs: DistrictConfigs = DistrictConfigs()

        # map polygons
        self.map_polygons: MapPolygons = map_polygons

        # cells
        self.cells: list[CellBase] = []
        self.n_cells: int = 0

    def run(self):
        """ Run segmentation. 
            to be implemented in child classes. 
        """
        raise NotImplementedError
