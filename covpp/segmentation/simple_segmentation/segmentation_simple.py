"""Simple segmentation method creating a single cell for the entire zone."""
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase


class Segmentation(SegmentationBase):
    """Single-cell segmentation plug-in.

    Steps:
        1. Mirror the outer workspace polygon as the only cell boundary.
        2. Skip adjacency creation because there is exactly one cell.

    Result:
        ``self.cells`` contains exactly one ``CellBase`` covering the full zone.
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationSimple")

    def run(self) -> None:
        """Execute simple segmentation by creating a single cell."""
        self.create_cells()

    def create_cells(self) -> list[CellBase]:
        """Instantiate a single ``CellBase`` for the whole zone polygon."""
        cell = CellBase(idx=0, poly=self.map_polygons.b_poly, resolution=self.zmd.resolution)
        self.cells.append(cell)
        self.n_cells = 1

    def set_cell_adjacents(self):
        """No-op because a lone cell has no neighbours."""
        return
