import numpy as np
from skimage.graph import route_through_array
from scripts.base.my_utils import MyPath, MyPoint
from scripts.district.zone_map_data import ZoneMapData

FREE = 255
OCCUPIED = 0


class ConnectPaths:
    """ Connect cell paths """
    c_paths_px: list[MyPath]

    def __init__(self, zone_map_data: ZoneMapData, paths_px: list[MyPath]):
        self.ns = "ConnectPaths"
        print(f"[{self.ns}] Step: connect paths")

        #
        self.zmd = zone_map_data
        self.c_paths_px = []

        #
        self.c_paths_px = self.connect_1(paths_px)

    def connect_1(self, paths_px: list[MyPath]) -> list[MyPath]:
        fp = FindPathInMap(self.zmd)
        c_paths = []
        for ip in range(len(paths_px) - 1):
            p0 = paths_px[ip].points[-1]
            p1 = paths_px[ip+1].points[0]
            # c_paths.append([p0, p1])
            c_paths.append(fp.find_path(start=p0, end=p1))
        return c_paths


class FindPathInMap:
    """ Find path between two points in map """
    occ_map: np.ndarray
    path_px: list[tuple[int, int]] | None

    def __init__(self, zmd: ZoneMapData):
        self.ns = "FindPathInMap"
        self.occ_map = zmd.map_ws
        self.s_x = zmd.rectangle.x_min
        self.s_y = zmd.rectangle.y_min

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
            print(f"[{self.ns}]: No path found from {start} to {end}")

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
            print(f"[{self.ns}]: Point {point} is not free, finding closest free cell")

        # find closest free cell
        y, x = np.where(occ_map == FREE)
        distances = np.sqrt((y - point[1])**2 + (x - point[0])**2)
        min_index = np.argmin(distances)
        return (x[min_index], y[min_index])


def get_start_end_points(paths_px: list[MyPath]) -> list:
    start_end_points: list[MyPoint] = []
    for path in paths_px:
        start_end_points.append((path[0], path[-1]))

    ends = [end.xy for _, end in start_end_points]
    starts = [start.xy for start, _ in start_end_points]
    ends = np.array(ends, dtype=np.int32)
    starts = np.array(starts, dtype=np.int32)
    return start_end_points
