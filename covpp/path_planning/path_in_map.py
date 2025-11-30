""" Module for finding paths in a map using different path planning methods. """
from importlib.metadata import entry_points
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.path_planning.path_planner_base import PathPlannerBase


class FindPathInMap:
    """ Class for finding path in a map using different path planning methods.

    Supported Methods:
        - "astar": A* algorithm using pyastar2d library
        - Additional methods can be added by implementing PathPlannerBase subclasses
    """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int]):
        self.lg = MyLogger("FindPathInMap")

        # ----------------------------------------------------------------------
        # create path planner
        # path planning method from configs
        pp_method = PipeLineConfig.pp_method
        planners = entry_points(group="path_planners")
        if pp_method not in planners.names:
            raise ValueError(f"Path planning method '{pp_method}' not recognized.")
        planner_cls = planners[pp_method].load()
        # ----------------------------------------------------------------------

        self.pp: PathPlannerBase = planner_cls(map_ws, g_xy)
