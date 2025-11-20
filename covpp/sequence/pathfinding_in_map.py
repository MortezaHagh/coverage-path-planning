
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.geometry import MyPath, MyPoint
# # path finding libraries
# # method 1
import pyastar2d
# # method 2
# from skimage.graph import route_through_array  # pylint: disable=E0611
# # method 3
# from pathfinding.core.diagonal_movement import DiagonalMovement
# from pathfinding.core.grid import Grid
# from pathfinding.finder.a_star import AStarFinder


class FindPathInMap:
    """ class for finding path between two points in map. """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int]):
        self.lg = MyLogger("FindPathInMap")

        # map - cost_map
        self.free_cost = 1
        self.occ_cost = 10
        binary_map = (map_ws == 255).astype(np.uint8)  # 1 = free, 0 = obstacle
        self.cost_map = np.where(binary_map == 1, 1, 10).astype(np.float32)   # method 1 # 1 = free, 1000 = obstacle
        # self.cost_map = np.where(binary_map == 1, 1, 0)                       # method 2 # 1 = free, 0 = obstacle
        # self.grid = Grid(matrix=self.cost_map)     # method 3 # grid for pathfinding algorithm (function find_path_al_pathfinding)

        # start point (offset in global map)
        self.s_x = g_xy[0]
        self.s_y = g_xy[1]

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
        path_px = self.find_path_al_pya(start, end)
        if path_px is None or len(path_px) == 0:
            self.lg.error(f"No path found from {start} to {end}")
            raise ValueError(f"No path found from {start} to {end}")

        # shift paths based on the zone map rectangle
        path_px = MyPath([MyPoint(p[0]+self.s_x, p[1]+self.s_y) for p in path_px])
        return path_px

    def find_path_al_pya(self, start: tuple[int, int], end: tuple[int, int]) -> list:
        """ use pyastar2d library to find path. """
        # swap x y to match the cost map
        start_s = (start[1], start[0])
        end_s = (end[1], end[0])
        # find path
        path = pyastar2d.astar_path(self.cost_map, start_s, end_s, allow_diagonal=False)
        path_list = [(x, y) for y, x in path]
        return path_list

    # def find_path_al_rout(self, start: tuple[int, int], end: tuple[int, int]) -> list:
    #     """ use skimage.graph.route_through_array to find path. """
    #     # swap x y to match the cost map
    #     start_s = (start[1], start[0])
    #     end_s = (end[1], end[0])
    #     # find path
    #     path_px, _ = route_through_array(self.cost_map, start_s, end_s, fully_connected=True)
    #     path_px = [(x, y) for y, x in path_px]  # swap back to x y
    #     return path_px

    # def find_path_al_pathfinding(self, start: tuple[int, int], end: tuple[int, int]) -> list:
    #     """ use pathfinding library to find path. """
    #     # start - end nodes
    #     start_n, end_n = self.grid.node(start[0], start[1]), self.grid.node(end[0], end[1])
    #     # find path
    #     finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    #     path, _ = finder.find_path(start_n, end_n, self.grid)
    #     path_list = [(x, y) for x, y in path]
    #     return path_list
