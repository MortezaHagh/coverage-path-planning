"""Base utilities for low-level path planners."""
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.geometry import MyPath, MyPoint


class PathPlannerBase:
    """Shared functionality for path planners that operate on cost maps.

    Process (implemented by subclasses):
        1. Prepare a cost map representation of the workspace via ``plan_path``.
        2. Convert global pixel coordinates into local planner coordinates.
        3. Run the concrete planner to obtain a list of pixels for the route.
        4. Lift the local result back into global coordinates and wrap it in ``MyPath``.

    Result:
        ``find_path`` returns a ``MyPath`` whose points live in the global map frame.
    """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int]):
        self.lg = MyLogger("PathPlannerBase")

        # map - cost_map
        self.free_cost = 1
        self.occ_cost = 100
        binary_map = (map_ws == 255).astype(np.uint8)  # 1 = free, 0 = obstacle
        self.cost_map = np.where(binary_map == 1, 1, 100).astype(np.float32)

        # start point (offset in global map)
        self.s_x = g_xy[0]
        self.s_y = g_xy[1]

    def plan_path(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        """Plan a local-frame path between two free pixels.

        Args:
        - start: Local pixel coordinate ``(x, y)`` for the start.
        - end: Local pixel coordinate ``(x, y)`` for the end.

        Notes:
        - Subclasses must replace this method with a concrete algorithm that consumes
            the prepared ``cost_map`` and returns an ordered list of local pixels.

        Returns:
        - list[tuple[int, int]]: Local-frame waypoints for the path.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def distance_between_points(self, p1: MyPoint, p2: MyPoint) -> float:
        """Return the travel distance by first computing the actual path.

        Steps:
            1. Invoke ``find_path`` which internally calls ``plan_path``.
            2. Convert the resulting ``MyPath`` into consecutive segments.
            3. Sum Euclidean segment lengths to obtain the travelled metric length.

        Returns:
            float: ``inf`` if no path exists, otherwise polyline length (pixels).
        """

        # get path
        path: MyPath = self.find_path(p1, p2)
        if path is None or len(path.points) == 0:
            return float('inf')

        # calculate path length
        length = np.sum(np.linalg.norm(np.diff(np.array(path.xy), axis=1), axis=1))
        return length

    def find_path(self, start: MyPoint, end: MyPoint) -> MyPath:
        """Compute a global-frame path between two map points.

        Args:
        - start: Global ``MyPoint`` start.
        - end: Global ``MyPoint`` end.

        Returns:
        - MyPath: Path with absolute pixel coordinates, ready for downstream consumers.

        Raises:
        - ValueError: If start/end are not free or if ``plan_path`` fails.
        """

        # start - end
        # shift to map coordinates
        start = (start.x - self.s_x, start.y - self.s_y)
        end = (end.x - self.s_x, end.y - self.s_y)

        # check if free, if not, find closest free cell
        if self.cost_map[start[1], start[0]] != self.free_cost or self.cost_map[end[1], end[0]] != self.free_cost:
            raise ValueError(f"Start {start} or end {end} point is not in free space.")

        # find path
        path_px = self.plan_path(start, end)
        if path_px is None or len(path_px) == 0:
            self.lg.error(f"No path found from {start} to {end}")
            raise ValueError(f"No path found from {start} to {end}")

        # shift paths based on the zone map rectangle
        path_px = MyPath([MyPoint(p[0]+self.s_x, p[1]+self.s_y) for p in path_px])
        return path_px
