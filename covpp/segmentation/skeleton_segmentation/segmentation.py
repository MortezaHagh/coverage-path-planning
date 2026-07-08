"""Voronoi-based segmentation for optimal workspace decomposition."""
import numpy as np
from shapely import Polygon
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.skeleton_segmentation.skeleton_decomposition import SkeletonDecomposition
from covpp.segmentation.segmentation_improvement.ebpts_graph_segmentation import EspecialBranchPointsSegmentation


class Segmentation(SegmentationBase):
    """ Segmentation class for workspace decomposition using skeletonization-based segmentation."""

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationSkeleton")

        self.polygons: list = [Polygon]                             # shapely polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray              # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray              # segmented workspace map with segmentation lines drawn

    def segment_map(self) -> None:
        """ Perform segmentation of the workspace map using skeletonization-based decomposition."""

        # skeleton decomposition
        skelet_de = SkeletonDecomposition(self.zmd)
        skelet_de.run()
        self.polygons = skelet_de.polygons
        self.adjacency_matrix = skelet_de.adjacency_matrix
        self.map_ws_segmented = skelet_de.map_ws_segmented
        # self.plot_segmentation()

    def post_process_segmentation(self) -> None:
        """ Post-process the segmentation results to fix potential issues and improve segmentation quality."""
        self.post_process_polygons()

        # self.zmd.map_ws_segmented = self.map_ws_segmented
        # improver = EspecialBranchPointsSegmentation(self.zmd, self.polygons)
        # if improver.run():
        #     print("Segmentation improved using Especial Branch Points Segmentation.")
        #     # update solution with improved polygons and adjacency matrix
        #     self.polygons = improver.polygons
        #     self.adjacency_matrix = improver.adjacency_matrix
        #     self.map_ws_segmented = improver.map_ws_segmented

        # self.post_process_polygons()

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

    def create_map_ws_segmented(self) -> None:
        """Create a segmented workspace map with segmentation lines drawn."""
        self.zmd.map_ws_segmented = self.map_ws_segmented

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
