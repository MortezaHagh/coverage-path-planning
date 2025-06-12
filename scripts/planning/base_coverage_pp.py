import os
import json
import numpy as np
from matplotlib import pyplot as plt
from scripts.base.my_utils import MyPoint, MyPath
from scripts.base.my_utils import get_n_colors_plt
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import SpaceConfigs, PlanningConfigs, DistrictConfgs


class CoveragePPBase:
    ns: str  # namespace for logging
    zmd: ZoneMapData  # zone data
    map_polygons: MapPolygons  # map polygons
    # paths
    f_path_px: MyPath
    f_path_m: MyPath
    paths_px: list[MyPath]
    paths_m: list[MyPath]
    c_paths_px: list[MyPath]
    c_paths_m: list[MyPath]
    n_paths: int = 0
    total_path_len_m: float

    def __init__(self, zone_map_data: ZoneMapData):
        self.ns = "CoveragePPBase"

        # zone data
        self.zmd = zone_map_data
        self.map_polygons = MapPolygons(self.zmd)

        # # path
        # final complete connected path
        self.f_path_px = None
        self.f_path_m = None
        # coverage paths
        self.paths_px = []
        self.paths_m = []
        # connection paths between cells
        self.c_paths_px = []
        self.c_paths_m = []

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

    def post_planning(self) -> None:
        self.n_paths = len(self.paths_px)
        self.calculate_paths_length()
        #
        self.plot_paths_px()
        self.save_data()
        # self.plot_paths_m()

    # ==================================================================================================================

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

    # ==================================================================================================================

    def plot_paths_px(self) -> None:
        """ Plot the calculated coverage paths and connections between them.
        """
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        #
        self.plot_coverage_paths(ax)
        self.plot_connection_paths(ax)
        # self.plot_connections_arrow(ax)

        #
        ns = self.dcfgs.covpp_method
        boust_path = os.path.join(self.zmd.path, ns)
        os.makedirs(boust_path, exist_ok=True)
        seq_method = self.dcfgs.cell_sequence_method
        fig_path = os.path.join(boust_path, f"paths_{seq_method}.png")
        fig.savefig(fig_path, format="png", dpi=300)
        # plt.show()

    def plot_coverage_paths(self, ax):
        n = self.n_paths
        colors = get_n_colors_plt(n)
        # fig, ax = plt.subplots(1, 1)
        x, y = self.map_polygons.b_poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=1)
        for obs in self.map_polygons.obs_polys:
            x, y = obs.exterior.xy
            ax.plot(x, y, color="black", linewidth=1)

        for i, path in enumerate(self.paths_px):
            if path is None:
                continue
            pp = np.array(path.xy, dtype=np.int32)
            ax.plot(pp[:, 0], pp[:, 1], color=colors[i], linewidth=2, linestyle='-')
            # start and end points
            ax.plot(pp[0, 0], pp[0, 1], 'o', color=colors[i], markersize=10)
            ax.text(pp[0, 0], pp[0, 1], f'S{i}', fontsize=9, color='k')
            ax.plot(pp[-1, 0], pp[-1, 1], 's', color=colors[i], markersize=10)
            ax.text(pp[-1, 0], pp[-1, 1], f'E{i}', fontsize=9, color='k')

    def plot_connection_paths(self, ax):
        for c_path in self.c_paths_px:
            if len(c_path.xy) < 2:
                continue
            pp = np.array(c_path.xy, dtype=np.int32)
            ax.plot(pp[0, 0], pp[0, 1], "o", color='gray', markersize=3)
            ax.plot(pp[-1, 0], pp[-1, 1], "o", color='gray', markersize=3)
            ax.plot(pp[:, 0], pp[:, 1], color='gray', linestyle=':')

    def save_data(self):
        len_m = self.total_path_len_m
        data = {"centers_seq_length": round(len_m, 2)}
        with open(self.zmd.result_file, 'w') as f:
            json.dump(data, f, indent=4)
            f.write("\n")

    def plot_paths_m(self):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("paths_m")
        #
        n = self.n_paths
        colors = get_n_colors_plt(n)
        for i, path in enumerate(self.paths_m):
            if path is None:
                continue
            pp = np.array(path.xy, dtype=np.float32)
            ax.plot(pp[:, 0], pp[:, 1], color=colors[i], linewidth=2, linestyle='-')
        #
        for c_path in self.c_paths_m:
            if c_path is None:
                continue
            pp = np.array(c_path.xy, dtype=np.float32)
            ax.plot(pp[:, 0], pp[:, 1], color='gray', linestyle=':')
        plt.show()
