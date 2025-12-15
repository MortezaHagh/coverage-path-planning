"""Path finding entry point

Factory that loads a concrete planner and exposes a consistent interface
to compute paths on a workspace map.

Steps:
- Read configured planner name from pipeline configs
- Load planner implementation via entry points
- Instantiate planner with workspace map and global offset

Result:
- ``FindPathInMap.pp`` holds the ready-to-use planner instance.
"""
from importlib.metadata import entry_points
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.path_planning.path_planner_base import PathPlannerBase


class FindPathInMap:
    """Create a configured path planner for a workspace map.

    Supported Methods:
    - "astar": A* algorithm using pyastar2d
    - Additional methods: implement a subclass of ``PathPlannerBase`` and register it.
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

    def find_path(self, start, end):
        """Delegate path finding to the configured planner.

        Args:
        - start: Global start as ``MyPoint``.
        - end: Global end as ``MyPoint``.

        Returns:
        - MyPath: Global-frame path constructed by the underlying planner.

        Notes:
        - Mirrors ``PathPlannerBase.find_path`` while keeping planner details opaque.
        """
        return self.pp.find_path(start, end)
