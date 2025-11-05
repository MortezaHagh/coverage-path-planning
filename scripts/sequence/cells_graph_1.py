""" Module for creating a graph of cells with distances and adjacency information. """
import heapq
import numpy as np
from scripts.base.logger import MyLogger
from scripts.segmentation.cell_processed import PCell
from scripts.sequence.connect_paths import ConnectPaths


class CellsGraph:
    """Graph representation of cells with distances and adjacency information. """

    def __init__(self, cells: list[PCell], con_path: ConnectPaths):
        self.lg = MyLogger("CellsGraph")
        #
        self.cells: list[PCell] = cells
        self.n_cells: int = len(cells)
        self.con_path = con_path
        #
        self.dists_m: np.ndarray = np.full((self.n_cells, self.n_cells), np.inf, dtype=float)   # Distance matrix between cells
        self.adj_inds: list[list[int]] | None = None                                            # Adjacency indices for each cell

        # Create adjacency distance matrix
        # self.create_distance_matrix_center_center()
        self.create_distance_matrix_by_path()
        self.find_all_distances()

    def create_distance_matrix_center_center(self):
        adj_inds = [[] for _ in range(self.n_cells)]
        dists_m = np.full((self.n_cells, self.n_cells), np.inf, dtype=float)
        for i in range(self.n_cells):
            cxy1 = self.cells[i].cxy
            for j in self.cells[i].adjacent_ids:
                adj_inds[i].append(j)
                cxy2 = self.cells[j].cxy
                dists_m[i, j] = np.linalg.norm(np.array(cxy1) - np.array(cxy2))
                dists_m[j, i] = dists_m[i][j]
        self.dists_m = dists_m
        self.adj_inds = adj_inds

    def create_distance_matrix_by_path(self):
        adj_inds = [[] for _ in range(self.n_cells)]
        dists_m = np.full((self.n_cells, self.n_cells), np.inf, dtype=float)

        for i, cell in enumerate(self.cells):
            for adj_id in cell.adjacent_ids:
                adj_cell = self.cells[adj_id]
                if adj_cell is None:
                    continue

                # cell is empty
                if cell.path_px is None or adj_cell.path_px is None:
                    self.lg.warn(f"Cell {cell.id} or adjacent cell {adj_cell.id} has no path_px.")
                    dist = self.con_path.update_connections(i, adj_id, None, None)
                else:
                    # find path between cell end point and adjacent cell start point
                    dist = self.con_path.update_connections(i, adj_id, cell.end_point, adj_cell.start_point)

                # Update distance matrix
                dists_m[i, adj_id] = dist

                # Add to adjacency indices
                adj_inds[i].append(adj_id)

        self.dists_m = dists_m
        self.adj_inds = adj_inds

    def find_all_distances(self):
        """Calculate all distances between cells using Dijkstra's algorithm. """
        for i in range(self.n_cells):
            for j in range(self.n_cells):
                if i == j:
                    continue
                # check if distance is already calculated
                if self.dists_m[i, j] != np.inf:
                    continue
                dist, _ = self.dijkstra(i, j)
                if dist is not None:
                    self.dists_m[i, j] = dist

    def get_distance(self, start_index, end_index):
        path_len, _ = self.dijkstra(start_index, end_index)
        return path_len

    def dijkstra(self, start_index, end_index):
        # Initialize distances with infinity
        distances = [float('inf')] * self.n_cells
        distances[start_index] = 0

        # Priority queue: (distance, cell_index)
        heap = [(0, start_index)]

        # To reconstruct the path
        previous = [None] * self.n_cells

        visited = set()

        while heap:
            current_dist, current_index = heapq.heappop(heap)

            # Skip if we've already found a better path
            if current_dist > distances[current_index]:
                continue

            # If we've reached the target
            if current_index == end_index:
                break

            visited.add(current_index)

            # Explore neighbors
            # current_cell: Cell = self.cells[current_index]
            for neighbor_index in self.adj_inds[current_index]:
                if neighbor_index in visited:
                    continue

                # Calculate distance
                new_dist = current_dist + self.dists_m[current_index, neighbor_index]

                # If found a shorter path to neighbor
                if new_dist < distances[neighbor_index]:
                    distances[neighbor_index] = new_dist
                    previous[neighbor_index] = current_index
                    heapq.heappush(heap, (new_dist, neighbor_index))

        # Reconstruct path if we found one
        if previous[end_index] is not None or end_index == start_index:
            path = []
            current = end_index
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
            return distances[end_index], path
        else:
            return None, None  # No path exists
