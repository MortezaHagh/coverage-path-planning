"""Voronoi-based segmentation for optimal workspace decomposition."""
import numpy as np
from shapely import Polygon
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.sweep_skeleton_decomposition.sweep_skeleton_decomposition import SweepSkeletonDecomposition


class Segmentation(SegmentationBase):
    """ Segmentation class for workspace decomposition using skeletonization-based segmentation."""

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationSkeleton")

        self.adjacency_matrix: np.ndarray = np.ndarray              # adjacency matrix between segments

    def segment_map(self) -> None:
        """ Perform segmentation of the workspace map using skeletonization-based decomposition."""

        # skeleton decomposition
        skelet_de = SweepSkeletonDecomposition(self.zmd)
        skelet_de.run()
        self.polygons = skelet_de.polygons
        self.adjacency_matrix = skelet_de.adjacency_matrix
        self.map_ws_segmented = skelet_de.map_ws_segmented
        # self.plot_segmentation()

    def create_cells(self) -> None:
        """Create CellBase objects for each segmented cell. """
        for i, poly in enumerate(self.polygons):
            cell = CellBase(i, poly, self.zmd.resolution)
            self.cells.append(cell)
        self.n_cells = len(self.cells)

    def set_cell_adjacents(self) -> None:
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
    #     for i, poly in enumerate(self.polygons):
    #         x, y = poly.exterior.xy
    #         ax.plot(x, y)
    #         # text
    #         ax.text(poly.centroid.x, poly.centroid.y, str(i), fontsize=12, ha='center')

    #     ax.set_title("Segmented Polygons")
    #     plt.show()
