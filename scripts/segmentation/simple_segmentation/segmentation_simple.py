""" Simple Segmentation Method."""
from scripts.base.logger import MyLogger
from scripts.segmentation.cell_base import CellBase
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.segmentation.segmentation_base import SegmentationBase


class Segmentation(SegmentationBase):
    """
    Interface for Simple segmentation method.
    Creates one cell for the whole zone area.
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg = MyLogger("SegmentationSimple")

    def run(self) -> None:
        """ Run simple segmentation method. """

        # create one cell for the whole area
        cell = CellBase(idx=0, poly=self.map_polygons.b_poly)
        self.cells.append(cell)
        self.n_cells = 1
