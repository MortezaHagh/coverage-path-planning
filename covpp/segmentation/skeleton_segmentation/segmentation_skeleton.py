"""Voronoi-based segmentation for optimal workspace decomposition."""
import numpy as np
from shapely import Polygon
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.skeleton_segmentation.skeleton_decomposition import SkeletonDecomposition


class Segmentation(SegmentationBase):
    """ """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationSkeleton")

        #
        self.polygons: list = [Polygon]                             # shapely polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray              # adjacency matrix between segments

    def run(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        skelet_de = SkeletonDecomposition(self.zmd)
        skelet_de.skeleton_decomposition()
        self.polygons = skelet_de.polygons
        self.adjacency_matrix = skelet_de.adjacency_matrix

        # create cells
        self.create_cells()
        self.set_cell_adjacents()

    def create_cells(self):
        """Create CellBase objects for each segmented cell. """
        for i, poly in enumerate(self.polygons):
            cell = CellBase(i, poly, self.zmd.resolution)
            self.cells.append(cell)
        self.n_cells = len(self.cells)

    def set_cell_adjacents(self):
        """Set adjacent cells for each cell based on shared boundaries or other criteria. """

        for i, cell in enumerate(self.cells):
            adjacent_indices = np.where(self.adjacency_matrix[i] == 1)[0]
            for j in adjacent_indices:
                cell.add_adjacent_id(j)

    # def plot_segmentation(self) -> None:
    #     """Plot segmentation results."""
    #     from matplotlib import pyplot as plt
    #     plt.close('all')
    #     fig, ax = plt.subplots()
    #     ax.set_aspect('equal')
    #     for poly in self.polygons:
    #         x, y = poly.exterior.xy
    #         ax.plot(x, y)

    #     ax.set_title("Segmented Polygons")
    #     plt.show()
