""" Coverage Path Planning - Local Energy Minimization Module """
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.configs import SpaceConfig
from covpp.segmentation.cell_processed import PCell


class GridNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.n_adj = 0          # number of adjacent nodes
        self.adj_nodes = []     # list of adjacent nodes
        self.free: bool = True  # is the node free or occupied

    def add_adj_node(self, node: 'GridNode'):
        self.adj_nodes.append(node)
        self.n_adj += 1


class PGrid:
    def __init__(self, rows: int, cols: int, grid_l: int, g_map: np.ndarray = None):
        self.map: np.ndarray = g_map
        self.rows = rows
        self.cols = cols
        self.nodes: np.ndarray = np.empty((rows, cols), dtype=object)
        self.current_node: GridNode = None
        self.heading: float = 0.0
        self.n = self.rows * self.cols              # total number of nodes
        self.n_occupied: int = 0                    # number of occupied nodes
        self.n_free: int = self.rows * self.cols    # number of free nodes
        self.grid_l: int = grid_l                   # size of each grid cell in pixels

    def set_n_occupied(self, n_occupied: int):
        self.n_occupied = n_occupied
        self.n_free = self.n - n_occupied


class GridGenerator:
    """ Creates a grid representation of the cell map for planning."""

    def __init__(self, cell: PCell, resolution: float):
        self.lg = MyLogger(f"GridGenerator-C{cell.id}")

        # coverage cell
        self.cell: PCell = cell

        # calculate grid parameters
        cleaning_r_px = SpaceConfig.cleaning_r / resolution
        grid_l = int(np.sqrt(2) * cleaning_r_px)

        # attributes
        self.grid_l = grid_l  # size of each grid cell in pixels
        self.grid: PGrid = None
        self.map: np.ndarray = None
        self.start_node: GridNode = None

        self.lg.info2("Creating grid representation of the cell map.")

        # create grid
        self.calculate_grid_size()
        self.create_grid()
        self.find_start_node()
        self.find_start_heading()
        self.grid.current_node = self.start_node
        self.grid.heading = self.heading

        # # visual debug info
        # import cv2 as cv
        # viz_map = cv.cvtColor(self.cell.map, cv.COLOR_GRAY2BGR)
        # cv.namedWindow("Cell Map", cv.WINDOW_NORMAL)
        # cv.imshow("Cell Map", viz_map)
        # cv.waitKey(0)

    def calculate_grid_size(self):
        if not self.cell.rect_like:
            return

        # calculate width and length of the cell
        x, y = self.cell.rectangle_poly.exterior.coords.xy
        points = np.array(list(zip(x, y)), dtype=np.int32)
        sides = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
        width = np.min(sides)
        # length = np.max(sides)

        # update grid_l based on smaller dimension
        n_rows = width // self.grid_l
        self.grid_l = int(np.ceil(width / n_rows))

    def create_grid(self) -> None:

        # ---------------------------------
        # create new downsampled map

        # cell map
        c_map = self.cell.map

        # Crop to multiple of cell_size for simplicity
        cell_size = self.grid_l
        h, w = c_map.shape
        h_new = h // cell_size
        w_new = w // cell_size
        map_cropped = c_map[:h_new * cell_size, :w_new * cell_size]

        # Reshape into blocks and apply max pooling
        map_blocks = map_cropped.reshape(h_new, cell_size, w_new, cell_size)
        g_map = map_blocks.max(axis=(1, 3)).astype(np.uint8)

        # ---------------------------------
        # create grid
        rows, cols = g_map.shape
        grid = PGrid(rows, cols, self.grid_l, g_map)

        # create nodes of the grid
        n_occupied = 0
        for i in range(rows):
            for j in range(cols):
                node = GridNode(j, i)
                # check if free or occupied
                if g_map[i, j] == 0:
                    node.free = False
                    n_occupied += 1
                grid.nodes[i, j] = node
        grid.set_n_occupied(n_occupied)

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
        self.map = g_map

    def find_start_node(self):

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
        self.start_node = start_node

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
