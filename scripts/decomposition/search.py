import heapq
from scripts.decomposition.cell import Cell


class SearchCells:
    def __init__(self, cells: list[Cell]):
        #
        self.cells = cells
        self.n_cells = len(cells)

        #
        self.path = None

        #
        start_index = 0
        end_index = self.n_cells - 1
        self.dijkstra(start_index, end_index)

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
            current_cell = self.cells[current_index]
            adjs = current_cell.
            for neighbor_index in current_cell.adjacent_indices:
                if neighbor_index in visited:
                    continue

                # Calculate distance (assuming uniform cost for simplicity)
                # You could modify this to use actual distances between coordinates
                new_dist = current_dist + 1  # or use Euclidean distance

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


# Example usage:
if __name__ == "__main__":
    # Example grid of cells
    cells = [
        Cell(0, 0, [1, 3]),       # Cell 0
        Cell(1, 0, [0, 2, 4]),     # Cell 1
        Cell(2, 0, [1, 5]),        # Cell 2
        Cell(0, 1, [0, 4, 6]),     # Cell 3
        Cell(1, 1, [1, 3, 5, 7]),  # Cell 4
        Cell(2, 1, [2, 4, 8]),     # Cell 5
        Cell(0, 2, [3, 7]),        # Cell 6
        Cell(1, 2, [4, 6, 8]),     # Cell 7
        Cell(2, 2, [5, 7])         # Cell 8
    ]

    start_index = 0  # Top-left corner
    end_index = 8    # Bottom-right corner

    distance, path = dijkstra(cells, start_index, end_index)

    if distance is not None:
        print(f"Shortest distance: {distance}")
        print("Path (cell indices):", path)
        print("Path coordinates:")
        for idx in path:
            cell = cells[idx]
            print(f"({cell.x}, {cell.y})")
    else:
        print("No path exists between the start and end cells.")
