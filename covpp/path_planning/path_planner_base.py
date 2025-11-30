""" Base class for path planners. """
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.geometry import MyPath, MyPoint


class PathPlannerBase:
    """ Base class for path planners. """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int]):
        self.lg = MyLogger("PathPlannerBase")

        # map - cost_map
        self.free_cost = 1
        self.occ_cost = 10
        binary_map = (map_ws == 255).astype(np.uint8)  # 1 = free, 0 = obstacle
        self.cost_map = np.where(binary_map == 1, 1, 10).astype(np.float32)

        # start point (offset in global map)
        self.s_x = g_xy[0]
        self.s_y = g_xy[1]

    def plan_path(self, start: MyPoint, end: MyPoint) -> MyPath:
        """ Plan path in map from start to end. """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def distance_between_points(self, p1: MyPoint, p2: MyPoint) -> float:
        """ Calculate distance between two points in the map.
            First find path, then calculate its length.
        """

        # get path
        path: MyPath = self.find_path(p1, p2)
        if path is None or len(path.points) == 0:
            return float('inf')

        # calculate path length
        length = np.sum(np.linalg.norm(np.diff(np.array(path.xy), axis=1), axis=1))
        return length

    def find_path(self, start: MyPoint, end: MyPoint) -> MyPath:
        """ Find path in map from start to end. """

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
