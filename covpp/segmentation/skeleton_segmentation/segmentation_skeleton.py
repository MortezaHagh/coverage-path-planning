"""Voronoi-based segmentation for optimal workspace decomposition."""
from shapely import Polygon
from matplotlib import pyplot as plt
from covpp.base.configs import PipeLineConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.voronoi_segmentation.cell import Cell
from covpp.segmentation.skeleton_segmentation.skeleton_decomposition import SkeletonDecomposition


class Segmentation(SegmentationBase):
    """ """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        super().__init__(zmd, map_polygons)
        self.lg.update_name("SegmentationSkeleton")

        # # attributes for segmentation
        # self.poly_start_idxs: list[int] = []                # start indices of the polygons in self.vor.points
        # self.poly_end_idxs: list[int] = []                  # end indices of the polygons in self.vor.points
        # self.critical_points: list[CriticalPoint] = []      # list of critical points found in the Voronoi decomposition
        # self.cp_base1_idxs: list[int] = []                  # indices of the first basis points in self.vor.points
        # self.cp_base2_idxs: list[int] = []                  # indices of the second basis points in self.vor.points
        # self.rings: list[Ring] = []                         # segments created from critical points

    def run(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        skelet_de = SkeletonDecomposition(self.zmd)
        skelet_de.skeleton_decomposition()
