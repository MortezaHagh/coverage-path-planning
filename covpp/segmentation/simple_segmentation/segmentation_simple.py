"""Simple segmentation method creating a single cell for the entire zone."""
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase


class Segmentation(SegmentationBase):
    """Simple segmentation implementation creating one cell for entire zone. Creates a single cell.

    This is the simplest segmentation strategy, treating the entire workspace
    as a single cell without further decomposition. Useful for small or
    convex workspaces.
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationSimple")

    def run(self) -> None:
        """Execute simple segmentation by creating a single cell."""
        self.create_cells()

    def create_cells(self) -> list[CellBase]:
        # create one cell for the whole area
        cell = CellBase(idx=0, poly=self.map_polygons.b_poly, resolution=self.zmd.resolution)
        self.cells.append(cell)
        self.n_cells = 1

    def set_cell_adjacents(self):
        # no adjacents in simple segmentation
        return
