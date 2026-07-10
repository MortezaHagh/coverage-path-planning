import numpy as np
from shapely import Polygon
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.distance_transform_segmentation.distance_tf_decomposition import DistanceTransformSegmentation


class Segmentation(SegmentationBase):
    """ Segmentation class for workspace decomposition using distance transform-based segmentation."""

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationDistanceTransform")

        self.adjacency_matrix: np.ndarray = np.ndarray              # adjacency matrix between segments

    def segment_map(self):
        dist_tf_decomp = DistanceTransformSegmentation(self.zmd)
        dist_tf_decomp.run()
        self.polygons = dist_tf_decomp.polygons
        self.adjacency_matrix = dist_tf_decomp.adjacency_matrix
        self.map_ws_segmented = dist_tf_decomp.map_ws_segmented

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
