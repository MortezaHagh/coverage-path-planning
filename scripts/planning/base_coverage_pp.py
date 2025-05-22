import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import SpaceConfigs, PlanningConfigs
from scripts.boustrophedon.decomposition_interface import DecompositionInterface


class CoveragePPBase:
    zmd: ZoneMapData  # zone data
    dec_interface: DecompositionInterface
    paths_px: list[list[tuple[int, int]]]
    paths_m: list[list[tuple[float, float]]]
    n_paths: int

    def __init__(self, zone_map_data: ZoneMapData):
        self.ns = "CoveragePPBase"

        # zone data
        self.zmd = zone_map_data

        # path
        self.dec_interface = None
        self.paths_px = []
        self.paths_m = []
        self.n_paths = 0

        # parameters and configurations
        self.scfgs = SpaceConfigs()
        self.pcfgs = PlanningConfigs()

        # ---------------------------------------------
        # robot params
        self.robot_width = self.scfgs.robot_radius
        self.min_overlap_w = self.pcfgs.min_path_overlap
        self.cleaning_width = 2*self.scfgs.cleaning_r
        # map params
        self.resolution = zone_map_data.reslolution
        # planning configs
        self.min_points_distance = self.pcfgs.min_wp_dist

    def plan(self):
        self.coverage_planning()
        self.post_planning()

    def coverage_planning(self):
        """
        Virtual method to be implemented by subclasses.
        """
        raise NotImplementedError("coverage_planning() must be implemented by subclasses.")

    def rotate_waypoints(self):
        rect_data = self.zmd.final_rect_data
        rot = rect_data["rot"]
        tl = self.zmd.final_rectangle.tl
        resol = self.zmd.reslolution
        x_0, y_0 = tl[0]*resol, tl[1]*resol  # first vertex (from which rot is calculated)
        # rotate
        new_paths = [[self.rotate_point(p[0], p[1], rot) for p in path_m] for path_m in self.paths_m]
        # translate
        new_paths = [[(p[0] + x_0, p[1] + y_0) for p in new_path] for new_path in new_paths]
        # update
        self.paths_m = new_paths

    def create_path_px(self):
        self.paths_px = [[self.point_m_to_px(p) for p in path_m] for path_m in self.paths_m]

    def rotate_point(self, x, y, rotation):
        x_r = x * np.cos(rotation) - y * np.sin(rotation)
        y_r = x * np.sin(rotation) + y * np.cos(rotation)
        return (x_r, y_r)

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        return (int(p[0] / self.resolution), int(p[1] / self.resolution))

    def post_planning(self):
        self.n_paths = len(self.paths_px)
