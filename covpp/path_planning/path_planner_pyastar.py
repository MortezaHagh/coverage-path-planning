
""" Path planner using A* algorithm with pyastar2d library. """
import pyastar2d
import numpy as np
from covpp.path_planning.path_planner_base import PathPlannerBase


class PathPlanner(PathPlannerBase):
    """ class for finding path between two points in map. """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int]):
        super().__init__(map_ws, g_xy)
        self.lg.update_name("PathPlannerAstar")

    def plan_path(self, start: tuple[int, int], end: tuple[int, int]) -> list:
        """ use pyastar2d library to find path. """
        # swap x y to match the cost map
        start_s = (start[1], start[0])
        end_s = (end[1], end[0])
        # find path
        path = pyastar2d.astar_path(self.cost_map, start_s, end_s, allow_diagonal=False)
        path_list = [(x, y) for y, x in path]
        return path_list
