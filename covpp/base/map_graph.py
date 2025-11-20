
from collections import deque
import numpy as np
from covpp.base.logger import MyLogger


class MapSearchGraph:
    """ Graph structure for map search operations."""

    def __init__(self, map_arr: np.ndarray, free_value: int = 255, ns=''):
        self.lg = MyLogger('MapSearchGraph-' + ns)

        binary_map = np.where(map_arr == free_value, 1, 0).astype(np.uint8)
        self.cost_map = binary_map
        self.free = 1  # FREE cell value in cost map

    def find_closest_free_cell(self, point: np.ndarray) -> np.ndarray:
        """ if point is in the free cell, return it, otherwise find the closest free cell. """

        # check if point is free
        x, y = point[0], point[1]
        if self.cost_map[y, x] == self.free:
            return np.array([x, y], dtype=int)
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
            if self.cost_map[y, x] == self.free:
                return np.array([x, y], dtype=int)
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= ny < max_y and 0 <= nx < max_x and (nx, ny) not in visited:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        self.lg.error(f"No free cell found near {point}")
        raise ValueError(f"No free cell found near {point}")
