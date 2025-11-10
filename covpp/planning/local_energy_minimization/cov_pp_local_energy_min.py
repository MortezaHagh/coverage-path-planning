""" Coverage Path Planning - Local Energy Minimization Module """
import cv2 as cv
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.my_utils import angle_diff, MyPoint, MyPath
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.voronoi_segmentation.cell import Cell
from covpp.planning.coverage_pp_base import CoveragePathPlannerBase


class GridNode:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.visited = False
        self.n_adj = 0          # number of adjacent nodes
        self.adj_nodes = []     # list of adjacent nodes
        self.free: bool = True  # is the node free or occupied

    def add_adj_node(self, node: 'GridNode'):
        self.adj_nodes.append(node)
        self.n_adj += 1


class Grid:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.nodes: np.ndarray = np.empty((rows, cols), dtype=object)


class CoveragePathPlanner(CoveragePathPlannerBase):
    """Coverage Path Planning - Local Energy Minimization Module

    Implements a local energy minimization strategy for generating coverage paths
    within a given cell.

    Result:
        Sets path_m and path_px attributes for the cell.
    """

    def __init__(self, cell: Cell, zmd: ZoneMapData):
        super().__init__(cell, zmd)

        self.lg = MyLogger(f"CovPPLocalEnergyMin-Z{zmd.ind}-C{cell.id}")

        # attributes
        self.grid: Grid = None
        self.map: np.ndarray = None
        self.start_node: GridNode = None
        self.heading: float = 0.0

        # start coverage path planning for the input cell and map
        self.coverage_planning()

    def coverage_planning(self):

        self.create_grid()
        self.start_node = self.find_start_node()
        self.find_start_heading()

        # plot for debugging (new_map, start_node, heading)
        vis_map = cv.cvtColor(self.map, cv.COLOR_GRAY2BGR)
        cv.circle(vis_map, (self.start_node.x, self.start_node.y), 3, (0, 0, 255), -1)
        line_length = 20
        end_x = int(self.start_node.x + line_length * np.cos(self.heading))
        end_y = int(self.start_node.y + line_length * np.sin(self.heading))
        cv.line(vis_map, (self.start_node.x, self.start_node.y), (end_x, end_y), (255, 0, 0), 2)
        # show the map
        # create a window
        cv.namedWindow("Grid Map", cv.WINDOW_NORMAL)
        cv.imshow("Grid Map", vis_map)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def create_grid(self) -> None:

        # ---------------------------------
        # create new downsampled map

        # cell map
        c_map = self.cell.map

        cell_size = 10  # size of each grid cell in pixels
        # Crop to multiple of cell_size for simplicity
        h, w = c_map.shape
        h_new = h // cell_size
        w_new = w // cell_size
        map_cropped = c_map[:h_new * cell_size, :w_new * cell_size]

        # Reshape into blocks and apply max pooling
        map_blocks = map_cropped.reshape(h_new, cell_size, w_new, cell_size)
        new_map = map_blocks.max(axis=(1, 3)).astype(np.uint8)

        # ---------------------------------
        # create grid
        rows, cols = new_map.shape
        grid = Grid(rows, cols)

        # create nodes of the grid
        for i in range(rows):
            for j in range(cols):
                node = GridNode()
                node.x = j
                node.y = i
                # check if free or occupied
                if new_map[i, j] == 0:
                    node.free = False
                grid.nodes[i, j] = node

        # ---------------------------------
        # set adjacent nodes
        for i in range(rows):
            for j in range(cols):
                node = grid.nodes[i, j]
                # check 8-connected neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbor = grid.nodes[ni, nj]
                            if neighbor.free:
                                node.add_adj_node(neighbor)

        self.grid = grid
        self.map = new_map

    def find_start_node(self) -> GridNode:

        # find first free node as starting point
        # expand from first node until first free node found
        start_node = None
        visited = set()
        queue = [(0, 0)]
        while queue:
            i, j = queue.pop(0)
            if (i, j) in visited:
                continue
            visited.add((i, j))
            node = self.grid.nodes[i, j]
            if node.free:
                start_node = node
                break
            # add neighbors to queue
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid.rows and 0 <= nj < self.grid.cols:
                        if (ni, nj) not in visited:
                            queue.append((ni, nj))

        if start_node is None:
            raise ValueError("No free start node found in the grid.")
        return start_node

    def find_start_heading(self):
        heading1 = (self.cell.major_direction + np.pi / 2) % (2 * np.pi)
        heading2 = (self.cell.major_direction - np.pi / 2) % (2 * np.pi)
        heading = heading1

        # select heading with more free space
        free_count1 = 0
        free_count2 = 0
        check_distance = 5  # number of nodes to check in each direction
        dists = np.arange(1, check_distance + 1)
        x1s = (self.start_node.x + dists * np.cos(heading1)).astype(int)
        y1s = (self.start_node.y + dists * np.sin(heading1)).astype(int)
        x2s = (self.start_node.x + dists * np.cos(heading2)).astype(int)
        y2s = (self.start_node.y + dists * np.sin(heading2)).astype(int)
        for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
            if 0 <= x1 < self.grid.cols and 0 <= y1 < self.grid.rows:
                if self.grid.nodes[y1, x1].free:
                    free_count1 += 1
            if 0 <= x2 < self.grid.cols and 0 <= y2 < self.grid.rows:
                if self.grid.nodes[y2, x2].free:
                    free_count2 += 1

        if free_count1 < free_count2:
            heading = heading2
        self.heading = heading
