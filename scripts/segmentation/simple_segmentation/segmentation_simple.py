"""Simple segmentation method creating a single cell for the entire zone."""
from scripts.base.logger import MyLogger
from scripts.segmentation.cell_base import CellBase
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.segmentation.segmentation_base import SegmentationBase


class Segmentation(SegmentationBase):
    """Simple segmentation implementation creating one cell for entire zone.

    This is the simplest segmentation strategy, treating the entire workspace
    as a single cell without further decomposition. Useful for small or
    convex workspaces.
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg = MyLogger("SegmentationSimple")

    def run(self) -> None:
        """Execute simple segmentation by creating a single cell."""

        # create one cell for the whole area
        cell = CellBase(idx=0, poly=self.map_polygons.b_poly)
        self.cells.append(cell)
        self.n_cells = 1
