import numpy as np
from skimage.graph import route_through_array  # pylint: disable=E0611
from scripts.base.logger import MyLogger
from scripts.planning.base_cell import BaseCell
from scripts.base.my_utils import MyPath, MyPoint
from scripts.district.zone_map_data import ZoneMapData

FREE = 255
OCCUPIED = 0


class ConnectPaths:
    """ Connect cell paths """
    zmd: ZoneMapData
    fp: 'FindPathInMap'
    all_connection_paths: np.ndarray  # 2D array of paths between cells
    all_distances: np.ndarray | None  # 2D array of distances between cells
    c_paths_px: list[MyPath]

    def __init__(self, zone_map_data: ZoneMapData):
        self.ns = "ConnectPaths"
        self.lg = MyLogger(self.ns)
        #
        self.fp = FindPathInMap(zone_map_data)

        #
        self.zmd = zone_map_data
        self.all_connection_paths = np.array([])
        self.all_distances = None
        self.c_paths_px = []

    def initialize(self, mcells: list[BaseCell]):
        n = len(mcells)
        self.all_connection_paths = np.full((n, n), None, dtype=object)
        self.all_distances = np.zeros((n, n), dtype=float)

    def find_all_connections(self, mcells: list[BaseCell]):
        self.lg.info2("Step: find all onnections paths")
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
        self.lg.info2("Step: find all distances")
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
            self.lg.warn("Calculating connections between paths, not using pre-calculated connections.")
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
    """ Find path between two points in map """
    occ_map: np.ndarray
    path_px: list[tuple[int, int]] | None

    def __init__(self, zmd: ZoneMapData):
        self.ns = "FindPathInMap"
        self.lg = MyLogger(self.ns)
        self.occ_map = zmd.map_ws
        self.s_x = zmd.rectangle.x_min
        self.s_y = zmd.rectangle.y_min

    def distance(self, p1: MyPoint, p2: MyPoint) -> float:
        path: MyPath = self.find_path(p1, p2)
        if path is None or len(path.points) == 0:
            return float('inf')
        # calculate path length
        length = 0.0
        for i in range(len(path.points) - 1):
            length += np.linalg.norm(np.array(path.points[i].xy) - np.array(path.points[i + 1].xy))
        return length

    def find_path(self, start: MyPoint, end: MyPoint) -> MyPath:
        """ Find path in map from start to end """

        # get map
        occ_map = self.occ_map
        binary_map = (occ_map == FREE).astype(np.uint8)  # 1 = free, 0 = obstacle

        # Invert to cost map: 1 (low cost) for free space, inf for obstacles
        cost_map = np.where(binary_map == 1, 1, np.inf)

        # start - end
        # shift to map coordinates
        start = (start.x - self.s_x, start.y - self.s_y)
        end = (end.x - self.s_x, end.y - self.s_y)  # shift to map coordinates
        start = self.find_closest_free_cell(start)
        end = self.find_closest_free_cell(end)

        # swap x y to match the cost map
        start = (start[1], start[0])
        end = (end[1], end[0])

        # find path
        path_px, _ = route_through_array(cost_map, start, end, fully_connected=True)

        # convert to list of tuples
        if not path_px:
            self.lg.error(f"No path found from {start} to {end}")

        # shift paths based on the zone map rectangle
        path_px = np.array(path_px) + np.array([self.s_y, self.s_x])
        path_px = MyPath([MyPoint(p[1], p[0]) for p in path_px])  # swap x y
        return path_px

    def find_closest_free_cell(self, point: tuple[int, int]) -> tuple[int, int]:
        """ if point is in the free cell, return it, otherwise find the closest free cell """
        # get map
        occ_map = self.occ_map

        # check if point is free
        x, y = point
        if occ_map[y, x] == FREE:
            return point
        else:
            self.lg.debug(f"Point {point} is not free, finding closest free cell")

        # find closest free cell
        y, x = np.where(occ_map == FREE)
        distances = np.sqrt((y - point[1])**2 + (x - point[0])**2)
        min_index = np.argmin(distances)
        return (x[min_index], y[min_index])
