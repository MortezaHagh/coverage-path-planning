"""Base class for coverage path planning algorithms.

This module defines the abstract interface that all coverage planning methods
must implement, enabling a plugin-based architecture.
"""
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.configs import PlanningConfig
from covpp.base.geometry import MyPoint, MyPath
from covpp.base.map_graph import MapSearchGraph
from covpp.segmentation.cell_planning import PCell
from covpp.district.zone_map_data import ZoneMapData
from covpp.path_planning.path_in_map import FindPathInMap


class CoveragePathPlannerBase:
    """Abstract base class for cell-level coverage path planning algorithms.

    All coverage planning implementations must inherit from this class and
    implement the coverage_planning() method. This design allows easy addition
    of new planning methods without modifying the core pipeline.

    Process:
        1. Initialize with a processed cell and zone map data.
        2. Implement coverage_planning() to generate a complete coverage path.
        3. Ensure paths are valid and within cell boundaries.
        4. Final validation of generated paths.

    Args:
        cell (PCell): Processed cell to plan coverage path for.
        zmd (ZoneMapData): Zone map data object.

    Attributes:
        cell (PCell): Target cell for path planning.
        zmd (ZoneMapData): Zone map data object.
        path_m (MyPath): Generated path in meters.
        path_px (MyPath): Generated path in pixels.

    Result:
        Sets path_m and path_px attributes for the cell.

    """

    def __init__(self, cell: PCell, zmd: ZoneMapData):
        self.lg = MyLogger("CoveragePlannerBase-Z" + str(zmd.ind) + f"-C{cell.id}")

        # zone map data
        self.zmd: ZoneMapData = zmd

        # configurations
        self.resolution: float = zmd.resolution

        # cell
        self.cell: PCell = cell

        # resulted paths
        self.path_m: MyPath = None      # path in meters
        self.path_px: MyPath = None      # path in pixels

    def run(self) -> None:
        """ Execute coverage path planning for the cell."""
        self.coverage_planning()  # perform coverage planning based on implemented method
        self.make_path_points_free()
        self.filter_path_by_distance()
        self.evaluate_path_edges()
        self.create_path_m()
        self.validate_paths()

    def coverage_planning(self) -> None:
        """Generate coverage path for the cell.

        This method must be implemented by all child classes. It should
        generate a complete coverage path for the cell and populate the
        path_px and path_m attributes.

        The implementation should:
            1. Generate path in pixels (path_px)
            2. Convert to meters using create_path_m()
            3. Handle any cell-specific constraints

        """
        raise NotImplementedError("Coverage planning subclasses must implement coverage_planning() method")

    # ==========================================================================
    # Path validation and adjustment
    # ==========================================================================

    def filter_path_by_distance(self) -> None:
        """Filter the generated path based on minimum and maximum waypoint distances."""
        if self.path_px:
            self.path_px = self.apply_distance_filter_path(self.path_px, self.resolution)

    def make_path_points_free(self) -> None:
        """ Ensure all points in the path are in free space. """
        mg = MapSearchGraph(self.cell.map, free_value=255, ns=f"{self.lg.name}")
        g_xy = self.cell.global_xy
        new_points = []
        xys = self.path_px.xy - g_xy  # local cell coordinates
        for xy in xys:
            free_p = mg.find_closest_free_cell(xy) + g_xy
            new_points.append(MyPoint(free_p[0], free_p[1]))
        self.path_px = MyPath(new_points)

    def evaluate_path_edges(self) -> None:
        """ Make sure all path edges are in free space. If not, reconnect them by finding path in the map."""

        # path finding in map object
        fp = FindPathInMap(self.cell.map, self.cell.global_xy).pp

        gxy = self.cell.global_xy
        points = self.path_px.points
        occupied = [self.cell.map[pt.y - gxy[1], pt.x - gxy[0]] != 255 for pt in points]

        # first and last points should be free
        if occupied[0] or occupied[-1]:
            raise ValueError("Path endpoints are not in free space.")

        # free indices
        free_indices = [i for i, occ in enumerate(occupied) if not occ]
        fi_1 = free_indices[:-1]
        fi_2 = free_indices[1:]

        # create new path
        new_points = []
        for i1, i2 in zip(fi_1, fi_2):
            if i2 - i1 > 1:
                # connect
                cpath = fp.find_path(start=points[i1], end=points[i2])
                cpath = self.apply_distance_filter_path(cpath, self.resolution)
                new_points.extend(cpath.points)
            else:
                new_points.append(points[i1])
                new_points.append(points[i2])
        self.path_px = MyPath(new_points)

    def validate_paths(self) -> bool:
        """Validate generated paths for the cell. """

        if not self.path_px or len(self.path_px.points) == 0:
            self.lg.warn(f"Cell {self.cell.id} has no path_px after coverage planning.")
            return False

        if not self.path_m or len(self.path_m.points) == 0:
            self.lg.warn(f"Cell {self.cell.id} has no path_m after coverage planning.")
            return False

        return True

    # ==========================================================================
    # Utility methods for path manipulation
    # ==========================================================================

    @staticmethod
    def apply_distance_filter_path(path: MyPath, resolution: float) -> MyPath:
        """ Filter the full path based on the minimum distance between waypoints. """
        min_wp_dist_px = PlanningConfig.min_wp_dist / resolution  # convert to px
        new_points = []
        prev_point = path.point(0)
        new_points.append(prev_point)
        for i in range(1, path.n):
            point = path.point(i)
            dist = np.sqrt((point.x - prev_point.x) ** 2 + (point.y - prev_point.y) ** 2)
            if dist >= min_wp_dist_px:
                new_points.append(point)
                prev_point = point

        # if distance between points is bigger than max_wp_dist_px, add as many points as needed
        max_wp_dist_px = PlanningConfig.max_wp_dist / resolution  # convert to px
        final_points = []
        prev_point = new_points[0]
        final_points.append(prev_point)
        for i in range(1, len(new_points)):
            point = new_points[i]
            dx = point.x - prev_point.x
            dy = point.y - prev_point.y
            dist = np.sqrt((dx) ** 2 + (dy) ** 2)
            if dist > max_wp_dist_px:
                n_points = int(np.ceil(dist / max_wp_dist_px))
                for j in range(1, n_points):
                    new_x = int(prev_point.x + dx * j / n_points)
                    new_y = int(prev_point.y + dy * j / n_points)
                    final_points.append(MyPoint(new_x, new_y))
            final_points.append(point)
            prev_point = point

        return MyPath(final_points)

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
