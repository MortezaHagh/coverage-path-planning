"""Base class for segmentation algorithms.

This module defines the abstract interface that all segmentation methods
must implement, enabling a plugin-based architecture.
"""
from abc import ABC, abstractmethod
import numpy as np
from shapely.geometry import Polygon
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.segmentation_improvement.ebpts_graph_segmentation import EspecialBranchPointsSegmentation


class SegmentationBase(ABC):
    """Abstract base class for zone segmentation algorithms.

    ``run`` is the template method used by the handler. Subclasses implement
    ``segment_map``, ``create_cells``, and ``set_cell_adjacents``; the base
    class performs polygon cleanup and creates the segmented-map result.

    Result:
        ``cells`` (list[CellBase]) describing the decomposition, ready to be
        wrapped into ``PCell`` objects by ``SegmentationHandler``.

    Args:
        zmd (ZoneMapData): Zone map data object.
        map_polygons (MapPolygons): Polygon representation of workspace and obstacles.

    Attributes:
        cells (list[CellBase]): List of cells created by segmentation.
        n_cells (int): Number of cells created.
        zmd (ZoneMapData): Zone map data object.
        map_polygons (MapPolygons): Polygon representation of the workspace.
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        self.lg = MyLogger("SegmentationBase")

        # zone map data
        self.zmd: ZoneMapData = zmd

        # map polygons
        self.map_polygons: MapPolygons = map_polygons

        # attributes
        self.map_ws_segmented: np.ndarray = np.ndarray      # map_ws with segmentation lines
        self.polygons: list[Polygon] = []
        self.cells: list[CellBase] = []
        self.n_cells: int = 0

    def run(self):
        """Run the standard segmentation lifecycle.

        The lifecycle is ``segment_map``, polygon post-processing,
        ``create_cells``, ``set_cell_adjacents``, and segmented-map creation.
        A plugin should override the individual hooks rather than this method
        unless it intentionally replaces the complete lifecycle.

        Plugins are expected to populate ``polygons`` and
        ``map_ws_segmented`` in ``segment_map``, ``cells`` and ``n_cells`` in
        ``create_cells``, and each cell's ``adjacent_ids`` in
        ``set_cell_adjacents``.
        """

        self.segment_map()
        self.post_process_segmentation()
        self.create_cells()
        self.set_cell_adjacents()
        self.create_map_ws_segmented()

    @abstractmethod
    def segment_map(self):
        """Populate raw cell polygons and the map showing segmentation lines."""
        raise NotImplementedError("Segmentation subclasses must implement segment_map() method")

    def post_process_segmentation(self):
        """Repair polygon geometry and apply the optional branch-point improvement."""
        self.post_process_polygons()

        # if not segmentation improvement:
        #     return

        self.zmd.map_ws_segmented = self.map_ws_segmented
        improver = EspecialBranchPointsSegmentation(self.zmd, self.polygons)
        if improver.run():
            print("Segmentation improved using Especial Branch Points Segmentation.")
            # update solution with improved polygons and adjacency matrix
            self.polygons = improver.polygons
            self.adjacency_matrix = improver.adjacency_matrix
            self.map_ws_segmented = improver.map_ws_segmented

        self.post_process_polygons()

    def post_process_polygons(self) -> None:
        """Repair invalid polygons and keep the largest part of a MultiPolygon."""
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

    @abstractmethod
    def create_cells(self):
        """Create one ``CellBase`` for each polygon and set ``n_cells``."""
        raise NotImplementedError("Segmentation subclasses must implement create_cells() method")

    @abstractmethod
    def set_cell_adjacents(self):
        """Populate each cell's adjacent-cell IDs."""
        raise NotImplementedError("Segmentation subclasses must implement set_cell_adjacents() method")

    def create_map_ws_segmented(self) -> None:
        """Create a map based on map_ws with segmentation lines drawn."""
        self.zmd.map_ws_segmented = self.map_ws_segmented
