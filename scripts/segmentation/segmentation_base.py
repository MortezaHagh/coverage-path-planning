"""Base class for segmentation algorithms.

This module defines the abstract interface that all segmentation methods
must implement, enabling a plugin-based architecture.
"""
from scripts.base.logger import MyLogger
from scripts.base.configs import PipeLineConfig
from scripts.segmentation.cell_base import CellBase
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData


class SegmentationBase:
    """Abstract base class for zone segmentation algorithms.

    All segmentation implementations must inherit from this class and
    implement the run() method. This design allows easy addition of
    new segmentation methods without modifying the core pipeline.

    Args:
        zmd: Zone map data object.
        map_polygons: Polygon representation of workspace and obstacles.

    Attributes:
        zmd: Zone map data object.
        map_polygons: Polygon representation of the workspace.
        cells: List of cells created by segmentation.
        n_cells: Number of cells created.
        dcfg: Pipeline configuration object.
        lg: Logger instance.

    Example:
        To create a new segmentation method:

        >>> class MySegmentation(SegmentationBase):
        ...     def run(self):
        ...         # Implement your segmentation logic
        ...         self.cells = [...]  # Create CellBase objects
        ...         self.n_cells = len(self.cells)
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        self.lg = MyLogger("SegmentationBase")

        # zone map data
        self.zmd: ZoneMapData = zmd

        # map polygons
        self.map_polygons: MapPolygons = map_polygons

        # configs
        self.dcfg: PipeLineConfig = PipeLineConfig()

        # cells
        self.cells: list[CellBase] = []
        self.n_cells: int = 0

    def run(self):
        """Execute the segmentation algorithm.

        This method must be implemented by all child classes. It should
        decompose the zone workspace into cells and populate the cells
        and n_cells attributes.

        Raises:
            NotImplementedError: If not implemented in child class.
        """
        raise NotImplementedError("Segmentation subclasses must implement run() method")

    def create_cells(self):
        """Create CellBase objects for each segmented cell.

        Raises:
            NotImplementedError: If not implemented in child class.
        """
        raise NotImplementedError("Segmentation subclasses must implement run() method")

    def set_cell_adjacents(self):
        """Set adjacent cells for each cell based on shared boundaries or other criteria.

        Raises:
            NotImplementedError: If not implemented in child class.
        """
        raise NotImplementedError("Segmentation subclasses must implement set_cell_adjacents() method")
