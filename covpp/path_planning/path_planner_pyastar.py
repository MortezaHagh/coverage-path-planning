
"""A*-based path planner plug-in backed by ``pyastar2d``."""
import pyastar2d
import numpy as np
from covpp.path_planning.path_planner_base import PathPlannerBase


class PathPlanner(PathPlannerBase):
    """Concrete ``PathPlannerBase`` that delegates to ``pyastar2d.astar_path``.

    Steps:
        1. Convert incoming (x, y) pixels into (row, col) pairs expected by
           ``pyastar2d``.
        2. Run grid-based A* on the prepared ``cost_map``.
        3. Convert the solver output back to the planner's coordinate frame.

    Result:
        ``plan_path`` returns a list of local pixel coordinates that ``find_path``
        turns into a global ``MyPath``.
    """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int]):
        super().__init__(map_ws, g_xy)
        self.lg.update_name("PathPlannerAstar")

    def plan_path(self, start: tuple[int, int], end: tuple[int, int]) -> list:
        """Compute a local path by invoking ``pyastar2d.astar_path``."""
        # swap x y to match the cost map
        start_s = (start[1], start[0])
        end_s = (end[1], end[0])
        # find path
        path = pyastar2d.astar_path(self.cost_map, start_s, end_s, allow_diagonal=False)
        path_list = [(x, y) for y, x in path]
        return path_list
