""" path planner using pyastar2d library. """
import numpy as np
# # path finding libraries
# # method 1
import pyastar2d
# # method 2
# from skimage.graph import route_through_array  # pylint: disable=E0611
# # method 3
# from pathfinding.core.diagonal_movement import DiagonalMovement
# from pathfinding.core.grid import Grid
# from pathfinding.finder.a_star import AStarFinder
from covpp.path_planning.path_planner_base import PathPlannerBase


class PathPlanner(PathPlannerBase):
    """ class for finding path between two points in map. """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int]):
        super().__init__(map_ws, g_xy)
        self.lg.update_name("PathPlannerTemp")

    def plan_path(self, start: tuple[int, int], end: tuple[int, int]) -> list:
        """ use pyastar2d library to find path. """
        # swap x y to match the cost map
        start_s = (start[1], start[0])
        end_s = (end[1], end[0])
        # find path
        path = pyastar2d.astar_path(self.cost_map, start_s, end_s, allow_diagonal=False)
        path_list = [(x, y) for y, x in path]
        return path_list
