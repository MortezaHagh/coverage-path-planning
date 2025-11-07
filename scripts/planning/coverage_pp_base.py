"""Base class for coverage path planning algorithms.

This module defines the abstract interface that all coverage planning methods
must implement, enabling a plugin-based architecture.
"""
import numpy as np
from scripts.base.logger import MyLogger
from scripts.base.my_utils import MyPoint, MyPath
from scripts.segmentation.cell_processed import PCell
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import PlanningConfig, PipeLineConfig


class CoveragePathPlannerBase:
    """Abstract base class for cell-level coverage path planning algorithms.

    All coverage planning implementations must inherit from this class and
    implement the coverage_planning() method. This design allows easy addition
    of new planning methods without modifying the core pipeline.

    Args:
        cell: Processed cell to plan coverage path for.
        zmd: Zone map data object.

    Attributes:
        cell: Target cell for path planning.
        zmd: Zone map data object.
        path_m: Generated path in meters (list of coordinate tuples).
        path_px: Generated path in pixels (list of coordinate tuples).
        resolution: Map resolution (meters per pixel).
        dcfg: Pipeline configuration object.
        pcfg: Planning configuration object.
        lg: Logger instance.

    Example:
        To create a new coverage planning method:

        >>> class MyPlanner(CoveragePathPlannerBase):
        ...     def coverage_planning(self):
        ...         # Implement your planning algorithm
        ...         self.path_px = [...]  # Generate path in pixels
        ...         self.create_path_m()  # Convert to meters
    """

    def __init__(self, cell: PCell, zmd: ZoneMapData):
        self.lg = MyLogger("CoveragePlannerBase-Z" + str(zmd.ind))

        # zone map data
        self.zmd: ZoneMapData = zmd

        # configurations
        self.dcfg = PipeLineConfig()
        self.pcfg = PlanningConfig()
        self.resolution: float = zmd.resolution

        # cell
        self.cell: PCell = cell

        # resulted paths
        self.path_m: list[tuple[float, float]] = []   # path in meters
        self.path_px: list[tuple[int, int]] = []      # path in pixels

    def coverage_planning(self) -> None:
        """Generate coverage path for the cell.

        This method must be implemented by all child classes. It should
        generate a complete coverage path for the cell and populate the
        path_px and path_m attributes.

        The implementation should:
            1. Generate path in pixels (path_px)
            2. Convert to meters using create_path_m()
            3. Handle any cell-specific constraints

        Raises:
            NotImplementedError: If not implemented in child class.
        """
        raise NotImplementedError("Coverage planning subclasses must implement coverage_planning() method")

    # ==========================================================================

    def create_path_m(self) -> None:
        """Convert path from pixels to meters using map resolution."""
        if self.path_px:
            self.path_m = MyPath([self.point_px_to_m(p) for p in self.path_px.points])

    def path_px_to_m(self, path_px: MyPath) -> MyPath:
        """Convert entire path from pixel coordinates to meters.

        Args:
            path_px: Path in pixel coordinates.

        Returns:
            Path in meter coordinates.
        """
        return MyPath([self.point_px_to_m(p) for p in path_px.points])

    def point_px_to_m(self, p: MyPoint) -> MyPoint:
        """Convert single point from pixel coordinates to meters.

        Args:
            p: Point in pixel coordinates.

        Returns:
            Point in meter coordinates.
        """
        return MyPoint(p.x * self.resolution, p.y * self.resolution)

    def rotate_path(self, path: MyPath, angle: float, origin: tuple) -> None:
        """Rotate entire path around an origin point.

        Args:
            path: Path to rotate.
            angle: Rotation angle in degrees.
            origin: Center point for rotation (x, y tuple).

        Returns:
            Rotated path.
        """
        return MyPath([self.rotate_point(p, angle, origin) for p in path.points])

    def rotate_point(self, p: MyPoint, angle: float, origin: tuple) -> MyPoint:
        """Rotate single point around an origin.

        Args:
            p: Point to rotate.
            angle: Rotation angle in radians.
            origin: Center point for rotation (x, y tuple).

        Returns:
            Rotated point with integer coordinates.
        """
        ox, oy = origin
        px, py = p.x, p.y

        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return MyPoint(int(qx), int(qy))
