import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import SpaceConfigs, PlanningConfigs, DistrictConfgs


class CoveragePPBase:
    ns: str  # namespace for logging
    zmd: ZoneMapData  # zone data
    #
    paths_px: list[list[tuple[int, int]]]
    paths_m: list[list[tuple[float, float]]]
    n_paths: int
    total_path_len_m: float

    def __init__(self, zone_map_data: ZoneMapData):
        self.ns = "CoveragePPBase"

        # zone data
        self.zmd = zone_map_data

        # path
        self.paths_px = []
        self.paths_m = []
        self.n_paths = 0
        self.total_path_len_m = 0.0

        # parameters and configurations
        self.scfgs = SpaceConfigs()
        self.dcfgs = DistrictConfgs()
        self.pcfgs = PlanningConfigs()

        # params
        self.cleaning_r = self.pcfgs.cleaning_r_ov
        self.cleaning_width = 2*self.pcfgs.cleaning_r_ov
        self.resolution = zone_map_data.reslolution
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

    def path_m_to_px(self, paths_m: list[tuple[float, float]]):
        return [[self.point_m_to_px(p) for p in path_m] for path_m in paths_m]

    def path_px_to_m(self, paths_px: list[tuple[int, int]]):
        return [[self.point_px_to_m(p) for p in path_px] for path_px in paths_px]

    def rotate_point(self, x, y, rotation):
        x_r = x * np.cos(rotation) - y * np.sin(rotation)
        y_r = x * np.sin(rotation) + y * np.cos(rotation)
        return (x_r, y_r)

    def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
        return (int(p[0] / self.resolution), int(p[1] / self.resolution))

    def point_px_to_m(self, p: tuple[int, int]) -> tuple[float, float]:
        return (p[0] * self.resolution, p[1] * self.resolution)

    def post_planning(self):
        self.n_paths = len(self.paths_px)
        self.calculate_paths_length()

    def calculate_paths_length(self):
        total_length = 0.0
        for path in self.paths_m:
            if len(path) < 2:
                continue
            for i in range(1, len(path)):
                p1 = path[i - 1]
                p2 = path[i]
                dist = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                total_length += dist
        self.total_path_len_m = round(total_length, 2)
