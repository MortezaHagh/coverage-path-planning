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

    def segment_map(self) -> None:
        """Execute simple segmentation by creating a single cell.

        Returns:
        - None
        """
        pass

    def post_process_segmentation(self) -> None:
        """No post-processing needed for single-cell segmentation.

        Returns:
        - None
        """
        pass

    def create_cells(self) -> None:
        """Instantiate a single ``CellBase`` for the whole zone polygon.

        Returns:
        - None
        """
        cell = CellBase(idx=0, poly=self.map_polygons.b_poly, resolution=self.zmd.resolution)
        self.cells.append(cell)
        self.n_cells = 1

    def set_cell_adjacents(self):
        """No-op because a lone cell has no neighbours."""
        return

    def create_map_ws_segmented(self) -> None:
        """Create a segmented map with a single cell.

        Returns:
        - None
        """
        pass
