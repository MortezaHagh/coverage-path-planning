"""Base class for segmentation algorithms.

This module defines the abstract interface that all segmentation methods
must implement, enabling a plugin-based architecture.
"""
from abc import ABC, abstractmethod

from covpp.base.logger import MyLogger
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.district.zone_map_data import ZoneMapData


class SegmentationBase(ABC):
    """Abstract base class for zone segmentation algorithms.

    Process expected from subclasses:
        1. Inspect ``self.zmd`` and ``self.map_polygons`` to decide how to split the zone.
        2. Populate ``self.cells`` with ``CellBase`` instances.
        3. Set adjacency information and ``self.n_cells`` for downstream planners.

    Result:
        ``cells`` (list[CellBase]) fully describing the decomposition ready to be
        wrapped into ``PCell`` objects.

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

        # cells
        self.cells: list[CellBase] = []
        self.n_cells: int = 0

    def run(self):
        """Execute the segmentation algorithm.

        Steps (expected from subclasses):
        - Generate raw cell polygons from workspace/obstacles
        - Call ``create_cells`` to build ``CellBase`` objects
        - Optionally call ``set_cell_adjacents`` and ``create_map_ws_segmented``

        Returns:
        - None
        """

        self.segment_map()
        self.post_process_segmentation()
        self.create_cells()
        self.set_cell_adjacents()
        self.create_map_ws_segmented()

        # raise NotImplementedError("Segmentation subclasses must implement run() method")

    @abstractmethod
    def segment_map(self):
        """Perform the segmentation of the workspace map.

        Returns:
        - None
        """
        raise NotImplementedError("Segmentation subclasses must implement segment_map() method")

    @abstractmethod
    def post_process_segmentation(self):
        """Post-process the segmented map.

        Returns:
        - None
        """
        raise NotImplementedError("Segmentation subclasses must implement post_process_segmentation() method")

    @abstractmethod
    def create_cells(self):
        """Create ``CellBase`` objects for each segmented cell.

        Returns:
        - None
        """
        raise NotImplementedError("Segmentation subclasses must implement run() method")

    @abstractmethod
    def set_cell_adjacents(self):
        """Set adjacency relationships between cells (shared boundaries, etc.).

        Returns:
        - None
        """
        raise NotImplementedError("Segmentation subclasses must implement set_cell_adjacents() method")

    @abstractmethod
    def create_map_ws_segmented(self) -> None:
        """Create a segmented workspace map with segmentation lines drawn.

        Returns:
        - None
        """
        raise NotImplementedError("Segmentation subclasses must implement create_map_ws_segmented() method")
