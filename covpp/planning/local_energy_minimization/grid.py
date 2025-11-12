""" Coverage Path Planning - Local Energy Minimization Module """
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.configs import SpaceConfig
from covpp.base.base_utils import angle_diff
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
    def __init__(self, rows: int, cols: int, l_row: int, l_col: int, g_map: np.ndarray = None):
        self.map: np.ndarray = g_map
        self.l_row: float = l_row                  # length of each row in pixels
        self.l_col: float = l_col                  # length of each column in pixels
        self.grid_l: float = min(l_row, l_col)     # size of each grid cell in pixels
        self.rows = rows
        self.cols = cols
        #
        self.current_node: GridNode = None
        self.current_heading: float = None
        #
        self.nodes: np.ndarray = np.empty((rows, cols), dtype=object)
        #
        self.n: int = self.rows * self.cols                 # total number of nodes
        self.n_occupied: int = np.sum(g_map == 0)           # number of occupied nodes
        self.n_free: int = self.n - self.n_occupied         # number of free nodes

    def set_n_occupied(self, n_occupied: int):
        self.n_occupied = n_occupied
        self.n_free = self.n - n_occupied


class GridGenerator:
    """ Creates a grid representation of the cell map for planning."""

    def __init__(self, cell: PCell, resolution: float):
        self.lg = MyLogger(f"GridGenerator-C{cell.id}")
        self.lg.info2("Creating grid representation of the cell map.")

        # coverage cell
        self.cell: PCell = cell

        # attributes
        self.grid: PGrid = None

        # create grid ---------------------------------
        grid_l = self.calculate_base_grid_size(resolution)

        if self.cell.rect_like:
            self.grid = self.create_grid_by_rectangle(grid_l)
        else:
            self.grid = self.create_grid_by_map(grid_l)

        # # for debugging
        # import cv2 as cv
        # # viz_map = cv.cvtColor(self.cell.map, cv.COLOR_GRAY2BGR)
        # # viz_map = cv.cvtColor(self.grid.map, cv.COLOR_GRAY2BGR)
        # viz_map = self.grid.map
        # cv.namedWindow("grid_map", cv.WINDOW_NORMAL)
        # cv.imshow("grid_map", viz_map)
        # cv.waitKey(0)

    def calculate_base_grid_size(self, resolution) -> int:

        cleaning_r_px = SpaceConfig.cleaning_r / resolution
        grid_l = int(np.sqrt(2) * cleaning_r_px)
        return grid_l

    def create_grid_by_rectangle(self, grid_l: int) -> PGrid:
        """ Create grid for rectangle-like cells. """
        rect_specs = self.calculate_grid_specs_rect(grid_l)
        grid = self.create_grid_and_nodes_rect(rect_specs)
        grid = self.add_adjacencies_rect(grid)
        grid.current_node = grid.nodes[0, 0]
        grid.current_heading = rect_specs["major_dir"]
        return grid

    def create_grid_by_map(self, grid_l: int) -> PGrid:
        """ Create grid from cell map by reshaping and reducing. """
        grid = self.create_grid_by_reshape_reduce_m(grid_l)
        grid = self.add_nodes_adjacents_m(grid)
        self.find_start_in_grid_m(grid)
        self.find_heading_in_grid_m(grid)
        return grid

    # --------------------------------------------------------------------------
    # Create grid from rectangle-like cell
    # --------------------------------------------------------------------------

    def create_grid_and_nodes_rect(self, rect_specs: dict) -> PGrid:
        """ Create grid and nodes for rectangle-like cells based on rectangle specs. """

        rs = rect_specs

        # create empty grid map
        g_map = np.full((rs["n_rows"], rs["n_cols"]), 255, dtype=np.uint8)

        # fill grid map by checking occupancy in each grid cell
        # create nodes of the grid
        nodes = np.empty((rs["n_rows"], rs["n_cols"]), dtype=object)
        theta = rect_specs["major_dir"]
        g_xy = self.cell.global_xy
        max_map_x = self.cell.map.shape[1]
        max_map_y = self.cell.map.shape[0]

        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
        for i in range(rs["n_rows"]):
            for j in range(rs["n_cols"]):
                # find center of the grid cell
                offset = np.array([j * rs["l_col"], i * rs["l_row"]])
                rotated_offset = rot_matrix @ offset
                x = int(rs["start_xy"][0] + rotated_offset[0])
                y = int(rs["start_xy"][1] + rotated_offset[1])
                nodes[i, j] = GridNode(x, y)

                # check occupancy of the point (x, y) in the cell map
                x_m, y_m = x - g_xy[0], y - g_xy[1]
                if 0 <= y_m < max_map_y and 0 <= x_m < max_map_x:
                    if self.cell.map[y_m, x_m] == 0:
                        g_map[i, j] = 0  # occupied
                        nodes[i, j].free = False
                else:
                    raise ValueError("Grid cell center is out of cell map bounds.")

        # # for debugging
        # # visualize self.cell.rectangle_poly.exterior.coords[:-1]
        # # and vizualize grid cell centers
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # rect_coords = np.array(self.cell.rectangle_poly.exterior.coords)
        # ax.plot(rect_coords[:, 0], rect_coords[:, 1], 'b-')
        # for i in range(rs["n_rows"]):
        #     for j in range(rs["n_cols"]):
        #         node = nodes[i, j]
        #         ax.plot(node.x, node.y, 'ro', markersize=2)
        # ax.set_aspect('equal')
        # plt.show()

        # create grid
        grid = PGrid(rs["n_rows"], rs["n_cols"], rs["l_row"], rs["l_col"], g_map)
        grid.nodes = nodes
        return grid

    def add_adjacencies_rect(self, grid: PGrid) -> PGrid:
        """ add 8 neighbors adjacents for rectangle-like grid. """
        node: GridNode
        n_rows = grid.nodes.shape[0]
        n_cols = grid.nodes.shape[1]
        for i in range(n_rows):
            for j in range(n_cols):
                node = grid.nodes[i, j]
                if node is None:
                    raise ValueError("Node is None when adding adjacents.")
                # Add 8-directional neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # Skip self
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n_rows and 0 <= nj < n_cols:
                            neighbor = grid.nodes[ni, nj]
                            if neighbor is not None:
                                node.add_adj_node(neighbor)
        return grid

    def calculate_grid_specs_rect(self, grid_l: int) -> dict:
        """ Calculate grid specifications for rectangle-like cells. """

        # scale rectangle 0.9 times to avoid boundary issues
        d_scale = -grid_l / 5
        scaled_rect = self.cell.rectangle_poly.buffer(d_scale, join_style=2, resolution=8)

        # start point of the rectangle
        coords = np.array(scaled_rect.exterior.coords[:-1])
        xy1 = coords[0]
        xy2 = coords[1]
        xy3 = coords[-1]

        # find width and length vectors
        vec_12 = xy2 - xy1
        vec_13 = xy3 - xy1
        l_12 = np.linalg.norm(vec_12)
        l_13 = np.linalg.norm(vec_13)
        th_12 = np.arctan2(vec_12[1], vec_12[0]) % (2 * np.pi)
        th_13 = np.arctan2(vec_13[1], vec_13[0]) % (2 * np.pi)
        if l_12 < l_13:
            width = l_12
            length = l_13
            th_l, th_w = th_13, th_12
        else:
            width = l_13
            length = l_12
            th_l, th_w = th_12, th_13

        # starting point
        start_xy = (int(xy1[0]), int(xy1[1]))

        # find number of rows and columns
        # calculate distance along each direction
        n_rows = int(np.ceil(width // grid_l))
        l_row = int(np.ceil(width / n_rows))
        n_rows += 1
        n_cols = int(np.ceil(length // grid_l))
        l_col = int(np.ceil(length / n_cols))
        n_cols += 1

        # directions (ccw or cw)
        angle_d = angle_diff(th_w, th_l)
        if angle_d < 0:
            l_row = -l_row

        # output dictionary
        rect_specs = {
            "width": width,
            "length": length,
            "start_xy": start_xy,
            "major_dir": th_l,
            "minor_dir": th_w,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "l_row": l_row,
            "l_col": l_col
        }
        return rect_specs

    # --------------------------------------------------------------------------
    # Create grid from map
    # --------------------------------------------------------------------------

    def create_grid_by_reshape_reduce_m(self, grid_l: int) -> PGrid:
        """ Create grid map by reshaping and reducing the cell map. """

        # Crop to multiple of cell_size for simplicity
        cell_size = grid_l
        h, w = self.cell.map.shape
        h_new = h // cell_size
        w_new = w // cell_size
        map_cropped = self.cell.map[:h_new * cell_size, :w_new * cell_size]

        # Reshape into blocks and apply max pooling
        map_blocks = map_cropped.reshape(h_new, cell_size, w_new, cell_size)
        g_map = map_blocks.max(axis=(1, 3)).astype(np.uint8)

        # create grid
        rows, cols = g_map.shape
        grid = PGrid(rows, cols, grid_l, grid_l, g_map)
        return grid

    def add_nodes_adjacents_m(self, grid: PGrid) -> PGrid:
        """ Add nodes to the grid (map based) and set their adjacencies. """

        # create nodes of the grid
        rows, cols = grid.map.shape
        n_occupied = 0
        for i in range(rows):
            for j in range(cols):
                x = j * grid.l_col + grid.l_col / 2
                y = i * grid.l_row + grid.l_row / 2
                node = GridNode(x, y)
                # check if free or occupied
                if grid.map[i, j] == 0:
                    node.free = False
                    n_occupied += 1
                grid.nodes[i, j] = node
        grid.set_n_occupied(n_occupied)

        # ---------------------------------
        # set adjacent nodes
        node: GridNode
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
        return grid

    def find_start_in_grid_m(self, grid: PGrid) -> None:
        """ Find first free node in the grid (map based) as starting point. """

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
            node = grid.nodes[i, j]
            if node.free:
                start_node = node
                break
            # add neighbors to queue
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid.rows and 0 <= nj < grid.cols:
                        if (ni, nj) not in visited:
                            queue.append((ni, nj))

        if start_node is None:
            raise ValueError("No free start node found in the grid.")
        grid.current_node = start_node

    def find_heading_in_grid_m(self, grid: PGrid) -> None:
        """ Find initial heading in the grid (map based) based on cell major direction. """

        heading1 = (self.cell.major_direction + np.pi / 2) % (2 * np.pi)
        heading2 = (self.cell.major_direction - np.pi / 2) % (2 * np.pi)
        heading = heading1

        # select heading with more free space
        free_count1 = 0
        free_count2 = 0
        check_distance = 5  # number of nodes to check in each direction
        dists = np.arange(1, check_distance + 1)
        x1s = (grid.current_node.x + dists * np.cos(heading1)).astype(int)
        y1s = (grid.current_node.y + dists * np.sin(heading1)).astype(int)
        x2s = (grid.current_node.x + dists * np.cos(heading2)).astype(int)
        y2s = (grid.current_node.y + dists * np.sin(heading2)).astype(int)
        for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
            if 0 <= x1 < grid.cols and 0 <= y1 < grid.rows:
                if grid.nodes[y1, x1].free:
                    free_count1 += 1
            if 0 <= x2 < grid.cols and 0 <= y2 < grid.rows:
                if grid.nodes[y2, x2].free:
                    free_count2 += 1

        if free_count1 < free_count2:
            heading = heading2
        grid.current_heading = heading
