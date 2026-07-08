import numpy as np
from shapely import Polygon
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.morphological_segmentation.morphological_decomposition import MorphologicalSegmentation


class Segmentation(SegmentationBase):
    """
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationMorphological")

        self.polygons: list = [Polygon]                             # shapely polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray              # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray              # segmented workspace map with segmentation lines drawn

    def segment_map(self):
        morph_decomp = MorphologicalSegmentation(self.zmd)
        morph_decomp.run()
        self.polygons = morph_decomp.polygons
        self.adjacency_matrix = morph_decomp.adjacency_matrix
        self.map_ws_segmented = morph_decomp.map_ws_segmented

    def post_process_segmentation(self):
        """Post-process the segmented map."""
        self.post_process_polygons()

    def post_process_polygons(self) -> None:
        """ Post-process polygons to fix potential issues such as MultiPolygons. """
        new_polys = []
        for poly in self.polygons:

            # fix potential invalid polygon issues
            poly = poly.buffer(0)

            # if it is multi polygon, take the largest one
            if poly.geom_type == 'MultiPolygon':
                self.lg.warn("Found MultiPolygon during post-processing. Taking the largest polygon.")
                poly = max(poly.geoms, key=lambda a: a.area)

            new_polys.append(poly)
        self.polygons = new_polys

    def create_cells(self):
        """Create ``CellBase`` objects for each segmented cell."""
        cells = []
        for idx, polygon in enumerate(self.polygons):
            cell = CellBase(idx, polygon, self.zmd.resolution)
            cells.append(cell)
        self.cells = cells
        self.n_cells = len(cells)

    def set_cell_adjacents(self):
        """Set adjacency relationships between cells (shared boundaries, etc.)."""
        for i, cell in enumerate(self.cells):
            for j, adjacent in enumerate(self.adjacency_matrix[i]):
                if adjacent:
                    cell.add_adjacent_id(j)

    def create_map_ws_segmented(self) -> None:
        """Create a map based on map_ws with segmentation lines drawn."""
        self.zmd.map_ws_segmented = self.map_ws_segmented
