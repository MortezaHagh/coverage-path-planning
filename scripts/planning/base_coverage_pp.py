import numpy as np
from scripts.base.my_utils import MyPoint, MyPath
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import SpaceConfigs, PlanningConfigs, DistrictConfgs


class CoveragePPBase:
    ns: str  # namespace for logging
    zmd: ZoneMapData  # zone data
    # paths
    f_path_px: MyPath
    f_path_m: MyPath
    paths_px: list[MyPath]
    paths_m: list[MyPath]
    n_paths: int = 0
    total_path_len_m: float

    def __init__(self, zone_map_data: ZoneMapData):
        self.ns = "CoveragePPBase"

        # zone data
        self.zmd = zone_map_data

        # path
        self.f_path_px = None  # final complete connected path in pixels
        self.f_path_m = None   # final complete connected path in meters
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

    def plan(self) -> None:
        self.coverage_planning()
        self.post_planning()

    def coverage_planning(self) -> None:
        """
        Virtual method to be implemented by subclasses.
        """
        raise NotImplementedError("coverage_planning() must be implemented by subclasses.")

    def rotate_waypoints(self) -> None:
        rect_data = self.zmd.final_rect_data
        rot = rect_data["rot"]
        tl = self.zmd.final_rectangle.tl
        resol = self.zmd.reslolution
        x_0, y_0 = tl[0]*resol, tl[1]*resol  # first vertex (from which rot is calculated)
        # rotate
        new_paths = [[self.rotate_point(p, rot) for p in path_m.points] for path_m in self.paths_m]
        # translate
        new_paths = [MyPath([MyPoint(p.x + x_0, p.y + y_0) for p in new_path]) for new_path in new_paths]
        # update
        self.paths_m = new_paths

    def path_m_to_px(self, paths_m: list[MyPath]) -> list[MyPath]:
        return [MyPath([self.point_m_to_px(p) for p in path_m.points]) for path_m in paths_m]

    def path_px_to_m(self, paths_px: list[MyPath]) -> list[MyPath]:
        return [MyPath([self.point_px_to_m(p) for p in path_px.points]) for path_px in paths_px]

    def rotate_point(self, p: MyPoint, rotation) -> MyPoint:
        x_r = p.x * np.cos(rotation) - p.y * np.sin(rotation)
        y_r = p.x * np.sin(rotation) + p.y * np.cos(rotation)
        return MyPoint(x_r, y_r)

    def point_m_to_px(self, p: MyPoint) -> MyPoint:
        return MyPoint(int(p.x / self.resolution), int(p.y / self.resolution))

    def point_px_to_m(self, p: MyPoint) -> MyPoint:
        return MyPoint(p.x * self.resolution, p.y * self.resolution)

    def post_planning(self) -> None:
        self.n_paths = len(self.paths_px)
        self.calculate_paths_length()

    def calculate_paths_length(self) -> None:
        total_length = 0.0
        for path in self.paths_m:
            if path.n < 2:
                continue
            for i in range(1, path.n):
                p1 = path.points[i - 1]
                p2 = path.points[i]
                dist = np.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
                total_length += dist
        self.total_path_len_m = round(total_length, 2)
