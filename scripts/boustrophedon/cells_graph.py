import heapq
import numpy as np
from scripts.boustrophedon.cell import Cell


class CellsGraph:
    def __init__(self, cells: list[Cell]):
        self.cells = cells
        self.n_cells = len(cells)
        self.dists_m = None
        self.adj_inds = None

        # Create adjacency distance matrix
        self.create_distance_matrix()

    def create_distance_matrix(self):
        n = self.n_cells
        adj_inds = [[] for _ in range(n)]
        dists_m = np.ones((n, n), dtype=float)*np.inf
        for i in range(n):
            cxy1 = self.cells[i].cxy
            for j in self.cells[i].adjacent_ids:
                adj_inds[i].append(j)
                cxy2 = self.cells[j].cxy
                dists_m[i, j] = np.linalg.norm(np.array(cxy1) - np.array(cxy2))
                dists_m[j, i] = dists_m[i][j]
        self.dists_m = dists_m
        self.adj_inds = adj_inds

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
                new_dist = current_dist + self.dists_m[current_index][neighbor_index]

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
