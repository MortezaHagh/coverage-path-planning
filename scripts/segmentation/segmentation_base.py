""" Base class for Segmentation."""
from scripts.base.logger import MyLogger
from scripts.base.configs import PipeLineConfig
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

        # map polygons
        self.map_polygons: MapPolygons = map_polygons

        # configs
        self.dcfgs: PipeLineConfig = PipeLineConfig()

        # cells
        self.cells: list[CellBase] = []
        self.n_cells: int = 0

    def run(self):
        """ Run segmentation.
            to be implemented in child classes.
        """
        raise NotImplementedError
