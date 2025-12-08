""" Generate grid representation of the cell map."""
import cv2 as cv
import numpy as np
from shapely.geometry import Point
from covpp.base.logger import MyLogger
from covpp.base.configs import SpaceConfig, PipeLineConfig
from covpp.base.geometry import angle_diff, find_longest_edge_poly
from covpp.segmentation.cell_planning import PCell
from covpp.district.zone_map_data import ZoneMapData


class GridNode:
    """ Containor for grid node information. """

    def __init__(self, x, y, i_row: int = None):
        self.x = x
        self.y = y
        self.i_row: int = i_row     # row index in the grid
        self.free: bool = True      # is the node free or occupied
        self.n_adj = 0              # number of adjacent nodes
        self.n_free_adj = 0         # number of free adjacent nodes
        self.adj_nodes = []         # list of adjacent nodes
        self.visited = False        # has the node been visited(set during path planning)

    def add_adj_node(self, node: 'GridNode'):
        """ add an adjacent node. """
        self.adj_nodes.append(node)
        self.n_adj += 1
        if node.free:
            self.n_free_adj += 1


class PGrid:
    """ Grid representation of the cell map for planning. """

    def __init__(self, l_row: int, l_col: int, g_map: np.ndarray):
        self.map: np.ndarray = g_map
        self.l_row: float = l_row                  # length of each row in pixels
        self.l_col: float = l_col                  # length of each column in pixels
        self.grid_l: float = min(l_row, l_col)     # size of each grid cell in pixels
        #
        self.current_node: GridNode = None
        self.current_heading: float = None
        #
        self.nodes: np.ndarray = np.empty(g_map.shape, dtype=object)
        #
        self.n: int = np.prod(g_map.shape)                  # total number of nodes
        self.n_occupied: int = np.sum(g_map == 0)           # number of occupied nodes
        self.n_free: int = self.n - self.n_occupied         # number of free nodes

    def set_n_occupied(self, n_occupied: int):
        """ Set number of occupied nodes and update free nodes. """
        self.n_occupied = n_occupied
        self.n_free = self.n - n_occupied


class GridGenerator:
    """ Creates a grid representation of the cell map for planning."""

    def __init__(self, cell: PCell, zmd: ZoneMapData):
        self.lg = MyLogger(f"GridGenerator-Z{zmd.ind}-C{cell.id}")
        self.lg.info2("Creating grid representation of the cell map.")

        # coverage cell
        self.cell: PCell = cell

        # attributes
        self.grid: PGrid = None

        # settings
        self.add_line_thresh = 0.2      # threshold to add extra boust line
        if PipeLineConfig.all_cleaned:
            self.add_line_thresh = 0.05

        # create grid ---------------------------------
        grid_l = self.calculate_base_grid_size(zmd.resolution)
        grid_l = int(1.0 * grid_l)  # reduce grid size a bit to avoid boundary issues

        if self.cell.rect_like:
            self.grid = self.create_grid_by_rectangle(grid_l)
        else:
            self.grid = self.create_grid_by_map(grid_l)

        self.plot_grid_map(zmd.segmentation_dir)

    def calculate_base_grid_size(self, resolution) -> float:
        """ Calculate base grid size based on cleaning radius and map resolution. """

        cleaning_r_px = SpaceConfig.cleaning_r / resolution
        # grid_l = np.sqrt(2) * cleaning_r_px
        grid_l = 2 * cleaning_r_px
        return grid_l

    def create_grid_by_rectangle(self, grid_l: int) -> PGrid:
        """ Create grid for rectangle-like cells. """
        rect_specs = self.calculate_grid_specs_rect(grid_l)
        grid = self.create_grid_and_nodes_rect(rect_specs)
        grid = self.add_adjacencies(grid)
        grid.current_node = grid.nodes[0, 0]
        grid.current_heading = rect_specs["major_dir"]
        return grid

    def create_grid_by_map(self, grid_l: int) -> PGrid:
        """ Create grid from cell map by reshaping and reducing. """
        grid = self.create_grid_m(grid_l)
        grid = self.add_adjacencies(grid)
        heading = (self.cell.major_direction + np.pi / 2) % (2 * np.pi)
        grid.current_heading = heading
        return grid

    # --------------------------------------------------------------------------
    # Create grid from rectangle-like cell
    # --------------------------------------------------------------------------

    def create_grid_and_nodes_rect(self, rect_specs: dict) -> PGrid:
        """ Create grid and nodes for rectangle-like cells based on rectangle specs. """

        rs = rect_specs

        # ----------------------------------------------------------------------
        # create meshgrid (in rectangle coordinates)
        y_coeff = 1 if rs["l_row"] > 0 else -1
        xs = np.linspace(0, rs["length"], rs["n_cols"])
        ys = np.linspace(0, rs["width"]*y_coeff, rs["n_rows"])
        xv, yv = np.meshgrid(xs, ys)
        xv_flattened = xv.flatten()
        yv_flattened = yv.flatten()
        points_rect = np.vstack((xv_flattened, yv_flattened)).T
        # rotate and translate to global coordinates
        theta = rect_specs["major_dir"]
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
        points_rotated = (rot_matrix @ points_rect.T).T
        points_global = points_rotated + rs["start_xy"]
        points_map = points_global - self.cell.global_xy
        # make sure points are integers
        points_map = points_map.astype(int)
        points_global = points_global.astype(int)
        # reshape back to grid
        points_map = points_map.reshape((rs["n_rows"], rs["n_cols"], 2))
        points_global = points_global.reshape((rs["n_rows"], rs["n_cols"], 2))

        # ----------------------------------------------------------------------
        # create empty grid map and nodes
        g_map = np.full((rs["n_rows"], rs["n_cols"]), 255, dtype=np.uint8)
        nodes = np.empty((rs["n_rows"], rs["n_cols"]), dtype=object)
        max_map_y, max_map_x = self.cell.map.shape
        for i in range(rs["n_rows"]):
            for j in range(rs["n_cols"]):
                nodes[i, j] = GridNode(int(points_global[i, j, 0]), int(points_global[i, j, 1]), i_row=i)

                # check occupancy of the point (x, y) in the cell map
                x_m = points_map[i, j, 0]
                y_m = points_map[i, j, 1]
                if 0 <= y_m < max_map_y and 0 <= x_m < max_map_x:
                    pass
                    # assume free if inside rectangle
                    # if self.cell.map[y_m, x_m] == 0:  # occupied
                    #     g_map[i, j] = 0
                    #     nodes[i, j].free = False
                else:
                    raise ValueError("Grid cell center is out of cell map bounds.")

        # create grid
        grid = PGrid(rs["l_row"], rs["l_col"], g_map)
        grid.nodes = nodes
        return grid

    def calculate_grid_specs_rect(self, grid_l: int) -> dict:
        """ Calculate grid specifications for rectangle-like cells. """

        # scale rectangle to avoid boundary issues
        d_scale = -grid_l / 10
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
            major_dir, minor_dir = th_13, th_12
        else:
            width = l_13
            length = l_12
            major_dir, minor_dir = th_12, th_13

        # starting point
        start_xy = (int(xy1[0]), int(xy1[1]))

        # find number of rows and columns
        # calculate distance along each direction
        n_rows = width / grid_l
        if n_rows % 1 > self.add_line_thresh:
            n_rows = int(np.ceil(n_rows))
        else:
            n_rows = int(n_rows)
        l_row = int(np.ceil(width / n_rows))
        n_rows += 1
        #
        n_cols = length / grid_l
        if n_cols % 1 > self.add_line_thresh:
            n_cols = int(np.ceil(n_cols))
        else:
            n_cols = int(n_cols)
        l_col = int(np.ceil(length / n_cols))
        n_cols += 1

        # directions (ccw or cw)
        angle_d = angle_diff(minor_dir, major_dir)
        if angle_d < 0:
            l_row = -l_row

        # output dictionary
        rect_specs = {
            "width": width,
            "length": length,
            "start_xy": start_xy,
            "major_dir": major_dir,
            "minor_dir": minor_dir,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "l_row": l_row,
            "l_col": l_col
        }
        return rect_specs

    # --------------------------------------------------------------------------
    # Create grid from map
    # --------------------------------------------------------------------------

    def create_grid_m(self, grid_l: int) -> PGrid:
        """ Create grid from cell map by generating a meshgrid and checking occupancy. """

        # find biggest edge of polygon
        poly = self.cell.poly
        scaled_poly_in = poly.buffer(-2, join_style=2, resolution=8)
        edge_props = find_longest_edge_poly(scaled_poly_in)
        p1 = edge_props['p1'] - self.cell.global_xy

        # scale rectangle to avoid boundary issues
        d_scale = grid_l / 2
        scaled_poly_out = poly.buffer(d_scale, join_style=2, resolution=8)

        # create a meshgrid centered at zero so 0 is guaranteed to be included
        n_h, n_w = self.cell.map.shape
        l_square = np.linalg.norm([n_h, n_w]) / np.sqrt(2)
        xs = np.arange(-l_square, l_square, grid_l)
        ys = np.arange(-l_square, l_square, grid_l)
        n_x, n_y = len(xs), len(ys)
        xv, yv = np.meshgrid(xs, ys)
        xv_flattened = xv.flatten()
        yv_flattened = yv.flatten()
        # points in the coordinates of the center of grid cells
        points_px = np.vstack((xv_flattened, yv_flattened)).T
        # rotate points based on major direction
        theta = (self.cell.major_direction + np.pi / 2) % (2 * np.pi)
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])

        # translate and rotate
        points_rotated = (rot_matrix @ points_px.T).T + np.array([n_w / 2, n_h / 2])

        # find closest point to p1 to align grid
        dists = np.linalg.norm(points_rotated - p1, axis=1)
        min_idx = np.argmin(dists)
        points_rotated = points_rotated - points_rotated[min_idx, :] + p1

        # Translate to global coordinates
        points_global = points_rotated + self.cell.global_xy
        points_map = points_rotated.astype(int)
        points_global = points_global.astype(int)

        # reshape back to grid
        points_map = points_map.reshape((n_y, n_x, 2))
        points_global = points_global.reshape((n_y, n_x, 2))

        # create empty grid map and nodes
        nodes: np.ndarray = np.empty((n_y, n_x), dtype=object)
        g_map: np.ndarray = np.full((n_y, n_x), 255, dtype=np.uint8)
        max_map_y, max_map_x = self.cell.map.shape
        start_node = None
        for i in range(n_y):
            for j in range(n_x):
                nodes[i, j] = GridNode(int(points_global[i, j, 0]), int(points_global[i, j, 1]), i_row=i)

                # check occupancy of the point (x, y) in the cell map
                x_m = points_map[i, j, 0]
                y_m = points_map[i, j, 1]
                if 0 <= y_m < max_map_y and 0 <= x_m < max_map_x:
                    if self.cell.map[y_m, x_m] == 0:  # occupied
                        # check if point is between the scaled rectangle and original rectangle (boundary area)
                        point = Point(points_global[i, j, 0], points_global[i, j, 1])
                        if scaled_poly_out.contains(point) and not scaled_poly_in.contains(point):
                            # between boundary area, consider free
                            pass
                        else:
                            g_map[i, j] = 0
                            nodes[i, j].free = False
                else:
                    g_map[i, j] = 0
                    nodes[i, j].free = False

                # set start node as first free node
                if start_node is None and nodes[i, j].free:
                    start_node = nodes[i, j]

        # create grid
        grid = PGrid(grid_l, grid_l, g_map)
        grid.nodes = nodes
        grid.current_node = start_node
        return grid

    # --------------------------------------------------------------------------
    # Add adjacencies
    # --------------------------------------------------------------------------

    def add_adjacencies(self, grid: PGrid) -> PGrid:
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

    # --------------------------------------------------------------------------
    # create grid from map by reshaping and reducing
    # --------------------------------------------------------------------------

    def create_grid_by_map_old(self, grid_l: int) -> PGrid:
        """ Create grid from cell map by reshaping and reducing. """
        grid = self.create_grid_by_reshape_reduce_m(grid_l)
        grid = self.add_nodes_adjacents_m(grid)
        self.find_start_in_grid_m(grid)
        self.find_heading_in_grid_m(grid)
        return grid

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
        grid = PGrid(grid_l, grid_l, g_map)
        return grid

    def add_nodes_adjacents_m(self, grid: PGrid) -> PGrid:
        """ Add nodes to the grid (map based) and set their adjacencies. """

        # create nodes of the grid
        rows, cols = grid.map.shape
        for i in range(rows):
            for j in range(cols):
                x = j * grid.l_col + grid.l_col / 2
                y = i * grid.l_row + grid.l_row / 2
                node = GridNode(x, y)
                # check if free or occupied
                if grid.map[i, j] == 0:
                    node.free = False
                grid.nodes[i, j] = node

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

    # --------------------------------------------------------------------------
    # Debugging plots
    # --------------------------------------------------------------------------

    def plot_grid_map(self, path: str) -> None:
        """ Plot grid map with start node and heading. """

        # plot map
        viz_map = cv.cvtColor(self.cell.map, cv.COLOR_GRAY2BGR)

        # plot nodes
        gxy = self.cell.global_xy
        rows, cols = self.grid.map.shape
        for i in range(rows):
            for j in range(cols):
                node = self.grid.nodes[i, j]
                if node.free:
                    cv.circle(viz_map, (int(node.x - gxy[0]), int(node.y - gxy[1])), 2, (0, 255, 0), -1)
                else:
                    cv.circle(viz_map, (int(node.x - gxy[0]), int(node.y - gxy[1])), 2, (0, 0, 255), -1)
        # plot start node and heading
        node = self.grid.current_node
        head = self.grid.current_heading
        gxy = self.cell.global_xy
        cv.circle(viz_map, (int(node.x - gxy[0]), int(node.y - gxy[1])), 3, (0, 0, 255), -1)
        x_end = int(node.x + 50 * np.cos(head) - gxy[0])
        y_end = int(node.y + 50 * np.sin(head) - gxy[1])
        cv.line(viz_map, (int(node.x - gxy[0]), int(node.y - gxy[1])), (x_end, y_end), (255, 0, 0), 2)

        # write image
        if PipeLineConfig.do_save_maps:
            cv.imwrite(f"{path}/grid_map_{self.cell.id}.png", viz_map)

        # # show image
        # cv.namedWindow("grid_map", cv.WINDOW_NORMAL)
        # cv.imshow("grid_map", viz_map)
        # cv.waitKey(0)

        # # plot using matplotlib
        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(figsize=(8, 8))
        # ax.set_aspect('equal', adjustable='box')

        # # plot map
        # ax.imshow(self.grid.map, cmap='gray', origin='upper')

        # # plot nodes
        # gxy = self.cell.global_xy
        # for node in self.grid.nodes.flatten():
        #     if node.free:
        #         ax.plot(node.x - gxy[0], node.y - gxy[1], 'go', markersize=2)
        #     else:
        #         ax.plot(node.x - gxy[0], node.y - gxy[1], 'ro', markersize=2)

        # # plot start node and heading
        # node = self.grid.current_node
        # head = self.grid.current_heading
        # gxy = self.cell.global_xy
        # ax.plot(node.x - gxy[0], node.y - gxy[1], 'bo', markersize=4)
        # x_end = node.x + 50 * np.cos(head) - gxy[0]
        # y_end = node.y + 50 * np.sin(head) - gxy[1]
        # ax.plot([node.x - gxy[0], x_end], [node.y - gxy[1], y_end], 'b-')
        # ax.set_title(f"Grid Map - Cell {self.cell.id}")
        # # fig.savefig(f"{path}/grid_map_{self.cell.id}.png")
        # plt.show()
        # plt.close(fig)
