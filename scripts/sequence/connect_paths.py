from collections import deque
import numpy as np

# # path finding libraries
# # method 1
import pyastar2d
# # method 2
# from skimage.graph import route_through_array  # pylint: disable=E0611
# # method 3
# from pathfinding.core.diagonal_movement import DiagonalMovement
# from pathfinding.core.grid import Grid
# from pathfinding.finder.a_star import AStarFinder
#
from scripts.base.logger import MyLogger
from scripts.base.my_utils import MyPath, MyPoint
from scripts.segmentation.cell_processed import PCell
from scripts.district.zone_map_data import ZoneMapData

FREE = 255
OCCUPIED = 0


class ConnectPaths:
    """ Connect cell paths.
    Find paths between end of one cell and start of the next cell in the map.
    Uses A* algorithm (skimage.graph.route_through_array) to find the shortest path in the occupancy grid.
    """

    def __init__(self, zone_map_data: ZoneMapData, ns: str = ""):
        self.lg = MyLogger(f"ConnectPaths-Z{zone_map_data.ind}-{ns}")

        # find path in map object
        self.fp: 'FindPathInMap' = FindPathInMap(zone_map_data)

        #
        self.zmd: ZoneMapData = zone_map_data
        self.all_connection_paths: np.ndarray = np.array([])    # 2D array of paths between cells
        self.all_distances: np.ndarray | None = None            # 2D array of distances between cells
        self.c_paths_px: list[MyPath] = []

    def initialize(self, mcells: list[PCell]):
        n = len(mcells)
        self.all_connection_paths = np.full((n, n), None, dtype=object)
        self.all_distances = np.zeros((n, n), dtype=float)

    def find_all_connections(self, mcells: list[PCell]):
        self.lg.info2("Step: find all connections paths")
        n = len(mcells)
        all_connections = np.full((n, n), None, dtype=object)

        for i in range(n):
            for j in range(i + 1, n):
                # if j in mcells[i].adjacent_cell_idxs:
                #     pass
                if mcells[i].end_point is None or mcells[j].start_point is None:
                    all_connections[i, j] = None
                    all_connections[j, i] = None
                else:
                    all_connections[i, j] = self.find_path(mcells[i].end_point, mcells[j].start_point)
                    all_connections[j, i] = self.find_path(mcells[j].end_point, mcells[i].start_point)
        self.all_connection_paths = all_connections

    def find_all_distances(self):
        self.lg.info2("find all distances")
        n = self.all_connection_paths.shape[0]
        all_distances = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                path: MyPath = self.all_connection_paths[i, j]
                if path is None or path.n == 0:
                    all_distances[i, j] = float('inf')
                else:
                    length = 0.0
                    p: MyPoint
                    for k, p in enumerate(path.points[:-1]):
                        length += np.linalg.norm(np.array(p.xy) - np.array(path.points[k+1].xy))
                    all_distances[i, j] = length
        self.all_distances = all_distances

    def connect_paths(self, paths_px: list[MyPath], seq_idxs: list[int] = None) -> list[MyPath]:
        self.lg.info2("Step: connecting paths")
        c_paths = []
        # if (not seq_idxs) or (self.all_connection_paths is None) or (self.all_connection_paths.shape[0] == 0):
        if not seq_idxs:
            # self.lg.info("Calculating connections between paths, not using pre-calculated connections.")
            for ip in range(len(paths_px) - 1):
                p0 = paths_px[ip].points[-1]
                p1 = paths_px[ip+1].points[0]
                c_paths.append(self.find_path(start=p0, end=p1))
        else:
            # if already calculated all connections and distances
            for ip, idx in enumerate(seq_idxs[:-1]):
                path = self.all_connection_paths[idx, seq_idxs[ip+1]]
                if path is None or len(path.points) == 0:
                    self.lg.warn(f"No connection path found between cells {idx} and {seq_idxs[ip+1]}. Creating...")
                    p0 = paths_px[ip].points[-1]
                    p1 = paths_px[ip+1].points[0]
                    path = self.find_path(start=p0, end=p1)
                c_paths.append(path)
        self.c_paths_px = c_paths

    def find_path(self, start: MyPoint, end: MyPoint) -> MyPath:
        return self.fp.find_path(start, end)

    def update_connections(self, start_idx: int, end_idx: int, start: MyPoint, end: MyPoint) -> MyPath:
        if start is None or end is None:
            self.lg.warn("Start or end index is None.")
            path = None
            length = float('inf')
            return
        path = self.fp.find_path(start, end)
        self.all_connection_paths[start_idx, end_idx] = path
        #
        length = 0.0
        p: MyPoint
        for k, p in enumerate(path.points[:-1]):
            length += np.linalg.norm(np.array(p.xy) - np.array(path.points[k+1].xy))
        self.all_distances[start_idx, end_idx] = length
        return length

# ======================================================================================================================


class FindPathInMap:
    """ class for finding path between two points in map. """

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("FindPathInMap")

        # map - cost_map
        occ_map: np.ndarray = zmd.map_ws
        binary_map = (occ_map == FREE).astype(np.uint8)  # 1 = free, 0 = obstacle
        self.cost_map = np.where(binary_map == 1, 1, 10).astype(np.float32)   # method 1 # 1 = free, 1000 = obstacle
        # self.cost_map = np.where(binary_map == 1, 1, 0)                       # method 2 # 1 = free, 0 = obstacle
        # self.grid = Grid(matrix=self.cost_map)     # method 3 # grid for pathfinding algorithm (function find_path_al_pathfinding)

        # and start point
        self.s_x = zmd.rectangle.x_min
        self.s_y = zmd.rectangle.y_min

    def distance_between_points(self, p1: MyPoint, p2: MyPoint) -> float:
        """ Calculate distance between two points in the map.
            First find path, then calculate its length.
        """

        # get path
        path: MyPath = self.find_path(p1, p2)
        if path is None or len(path.points) == 0:
            return float('inf')

        # calculate path length
        length = 0.0
        for i in range(len(path.points) - 1):
            length += np.linalg.norm(np.array(path.points[i].xy) - np.array(path.points[i + 1].xy))
        return length

    def find_path(self, start: MyPoint, end: MyPoint) -> MyPath:
        """ Find path in map from start to end. """

        # start - end
        # shift to map coordinates
        start = (start.x - self.s_x, start.y - self.s_y)
        end = (end.x - self.s_x, end.y - self.s_y)
        # check if free, if not, find closest free cell
        start = self.find_closest_free_cell(start)
        end = self.find_closest_free_cell(end)

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
        path = pyastar2d.astar_path(self.cost_map, start_s, end_s, allow_diagonal=True)
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

    def find_closest_free_cell(self, point: tuple[int, int]) -> tuple[int, int]:
        """ if point is in the free cell, return it, otherwise find the closest free cell. """

        # check if point is free
        x, y = point
        if self.cost_map[y, x] == 1:  # FREE
            return point
        else:
            self.lg.debug(f"Point {point} is not free, finding closest free cell")

        # Use BFS to find the nearest free cell for efficiency

        max_y, max_x = self.cost_map.shape
        visited = set()
        queue = deque()
        queue.append((point[0], point[1]))
        visited.add((point[0], point[1]))

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            x, y = queue.popleft()
            if self.cost_map[y, x] == 1:  # FREE
                return (x, y)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < max_y and 0 <= nx < max_x and (nx, ny) not in visited:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        self.lg.error(f"No free cell found near {point}")
        raise ValueError(f"No free cell found near {point}")

        # # find closest free cell
        # y, x = np.where(self.cost_map == 0)
        # # calculate distances to all free cells
        # distances = np.sqrt((y - point[1])**2 + (x - point[0])**2)
        # min_index = np.argmin(distances)
        # return (x[min_index], y[min_index])
