"""Base class for segmentation algorithms.

This module defines the abstract interface that all segmentation methods
must implement, enabling a plugin-based architecture.
"""
from covpp.base.logger import MyLogger
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.map_polygons import MapPolygons
from covpp.district.zone_map_data import ZoneMapData


class SegmentationBase:
    """Abstract base class for zone segmentation algorithms. Results in cells.

    All segmentation implementations must inherit from this class and
    implement the run() method. This design allows easy addition of
    new segmentation methods without modifying the core pipeline.

    Args:
        zmd (ZoneMapData): Zone map data object.
        map_polygons (MapPolygons): Polygon representation of workspace and obstacles.

    Attributes:
        zmd (ZoneMapData): Zone map data object.
        map_polygons (MapPolygons): Polygon representation of the workspace.
        cells (list[CellBase]): List of cells created by segmentation.
        n_cells (int): Number of cells created.
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

        This method must be implemented by all child classes. It should
        decompose the zone workspace into cells and populate the cells
        and n_cells attributes.
        """
        raise NotImplementedError("Segmentation subclasses must implement run() method")

    def create_cells(self):
        """Create CellBase objects for each segmented cell. """
        raise NotImplementedError("Segmentation subclasses must implement run() method")

    def set_cell_adjacents(self):
        """Set adjacent cells for each cell based on shared boundaries or other criteria. """
        raise NotImplementedError("Segmentation subclasses must implement set_cell_adjacents() method")
