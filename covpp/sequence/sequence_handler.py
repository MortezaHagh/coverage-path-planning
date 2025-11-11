"""Sequence optimization module for cell ordering and path connection.

This module handles the sequencing of coverage cells, connection of individual
cell paths, and generation of the complete zone coverage path.
"""
from importlib.metadata import entry_points
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from covpp.segmentation.cell_processed import PCell
from covpp.segmentation.map_polygons import MapPolygons
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.connect_paths import ConnectPaths
from covpp.sequence.sequence_optimizers import BaseSeqOpt
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig, PlanningConfig
from covpp.base.base_utils import MyPoint, MyPath
from covpp.base.base_utils import get_n_colors_cv, get_n_colors_plt


class SequencePaths:
    """Container for storing all path variants (coverage, connection, and full paths).

    Attributes:
        paths_px: List of coverage paths in pixels.
        paths_m: List of coverage paths in meters.
        c_paths_px: List of connection paths between cells in pixels.
        c_paths_m: List of connection paths between cells in meters.
        f_path_px: Final complete connected path in pixels.
        f_path_m: Final complete connected path in meters.
        n_paths: Number of coverage paths.
    """

    def __init__(self):
        self.paths_px: list[MyPath] = []
        self.paths_m: list[MyPath] = []
        self.c_paths_px: list[MyPath] = []
        self.c_paths_m: list[MyPath] = []
        self.f_path_px: MyPath = None
        self.f_path_m: MyPath = None
        self.n_paths: int = 0


class SequenceHandler:
    """Optimize cell visit order and create complete connected coverage path.

    This class handles the sequencing problem (determining optimal cell visit
    order) and path connection (creating transitions between cells) to generate
    a complete coverage path for the zone.

    Args:
        zmd: Zone map data object.

    Attributes:
        zmd: Zone map data object.
        cells: List of cells to sequence.
        n_cells: Number of cells.
        map_polygons: Polygon representation of workspace.
        connections: Path connection handler.
        paths: SequencePaths object containing all path variants.
        seq_idxs: Optimized sequence of cell indices.
        s_cells: Sorted cells according to optimized sequence.
        plotter: Visualization handler.
        pcfg: Planning configuration.
        dcfg: Pipeline configuration.

    Pipeline Steps:
        1. Optimize cell visit sequence
        2. Connect paths between consecutive cells
        3. Apply distance filtering and smoothing
        4. Generate complete path
        5. Update visualization maps
    """

    def __init__(self, zmd: ZoneMapData) -> None:

        self.lg = MyLogger("SequenceHandler")

        # attributes
        self.zmd: ZoneMapData = zmd
        self.map_polygons: MapPolygons = None
        #
        self.cells: list[PCell] = []
        self.n_cells: int = 0

        # configurations
        self.pcfg: PlanningConfig = PlanningConfig()
        self.dcfg: PipeLineConfig = PipeLineConfig()
        self.colors: list[tuple] = []

        # params
        self.min_points_distance: float = self.pcfg.min_wp_dist
        self.resolution: float = zmd.resolution

        # sequence optimizer
        sequencer_name = self.dcfg.sequence_method
        sequencers = entry_points(group="sequencers")
        if sequencer_name not in sequencers.names:
            raise ValueError(f"Sequence optimizer '{sequencer_name}' not recognized.")
        self.sequencer: BaseSeqOpt = sequencers[sequencer_name].load()()

        # connect paths interface
        self.connections: ConnectPaths = ConnectPaths(self.zmd)

        # # paths
        # final complete connected path
        self.f_path_px: MyPath = None
        self.f_path_m: MyPath = None
        # coverage paths
        self.paths_px: list[MyPath] = []
        self.paths_m: list[MyPath] = []
        # connection paths between cells
        self.c_paths_px: list[MyPath] = []
        self.c_paths_m: list[MyPath] = []

        # animation
        self.anim_line = None  # for animation
        self.anim = None  # for animation

        # results
        self.s_cells: list[PCell] = []
        self.seq_idxs: list[int] = []
        self.paths: SequencePaths = SequencePaths()

    def run(self, cells: list[PCell], map_polygons: MapPolygons) -> None:
        """ Run sequence creation and path connection. """
        self.cells = cells
        self.n_cells = len(cells)
        self.map_polygons = map_polygons

        # **********************
        self.create_sequence()
        self.connect_paths()
        self.post_planning()
        self.set_sequence_paths()
        self.plot_paths()
        self.update_map_planning()
        # **********************

    def create_sequence(self) -> None:
        """
        Create optimum sequence of cells to be covered.

        Steps:
        1. Calculate distances between cells (if needed).
        2. Optimize sequence of cells based on distances.
        3. Store paths in sequence paths.
        """
        self.lg.info1("Step: creating optimized cell sequence")

        # Initialize connections object with cells
        self.connections.initialize(self.cells)
        distances = None

        # Initialize distances between cells (paths start and end points)
        if self.dcfg.sequence_method == "milp":
            # calculate all paths and calculate distances between cells entries (based on cell paths)
            self.connections.find_all_connections(self.cells)
            self.connections.find_all_distances()
            distances = self.connections.all_distances

            # # create cells graph and distances
            # cg = CellsGraph(self.cells, self.connections)
            # distances = cg.dists_m

        # create sequence
        self.optimize(distances=distances)
        self.paths_m = [cell.path_m for cell in self.s_cells if cell.do_planning]
        self.paths_px = [cell.path_px for cell in self.s_cells if cell.do_planning]

    def optimize(self, distances: np.ndarray = None) -> None:
        if len(self.cells) == 0:
            return
        elif len(self.cells) == 1:
            self.s_cells = self.cells
            return
        # run sequence optimizer
        self.sequencer.run(cells=self.cells, distances=distances, start_idx=self.dcfg.start_cell_ind)
        self.seq_idxs = self.sequencer.tour_idxs
        self.s_cells = [self.cells[i] for i in self.seq_idxs]

    def connect_paths(self) -> None:
        """ Connect the paths based on the created sequence.
            connect end-start points of paths.
        """
        self.connections.connect_paths(self.paths_px, self.seq_idxs)
        self.c_paths_px = self.connections.c_paths_px
        self.c_paths_m = [self.path_px_to_m(path_px) for path_px in self.c_paths_px]

# Post Planning ====================================================================================================

    def post_planning(self) -> None:
        if len(self.paths_px) == 0 or len(self.paths_m) == 0:
            self.lg.error("No paths found. Cannot post_planning.")
            return

        self.apply_distance_filter_all_paths()
        self.create_full_path()

    def create_full_path(self) -> None:
        """ Create the full path by combining coverage paths and connection paths. """
        n_paths = len(self.paths_px)
        f_path_px = []
        for i in range(n_paths-1):
            f_path_px.extend(self.paths_px[i].points)
            f_path_px.extend(self.c_paths_px[i].points)
        f_path_px.extend(self.paths_px[-1].points)
        #
        self.f_path_px = MyPath(f_path_px)
        self.f_path_m = self.path_px_to_m(self.f_path_px)

    def plot_paths(self):
        self.plot_paths_px()
        self.plot_final_path_m()
        if self.pcfg.animation:
            self.animate_robot_paths()

    def set_sequence_paths(self) -> None:
        self.paths.paths_px = self.paths_px
        self.paths.paths_m = self.paths_m
        self.paths.c_paths_px = self.c_paths_px
        self.paths.c_paths_m = self.c_paths_m
        self.paths.f_path_px = self.f_path_px
        self.paths.f_path_m = self.f_path_m
        self.paths.n_paths = len(self.paths_px)

    # update map planning ==============================================================================================

    def update_map_planning(self) -> None:
        """ update map_planning with paths and cells, save map_planning. """
        if not self.dcfg.save_map_planning:
            return
        n = self.n_cells
        self.colors = get_n_colors_cv(n)
        self.add_cells_to_map_planning()
        self.add_cell_paths_to_map_planning()

    def add_cells_to_map_planning(self) -> None:
        for i, cell in enumerate(self.cells):
            pp = cell.points
            pp = np.array(pp, dtype=np.int32)  # Ensure pp is a numpy array of integers
            # add cells to map_planning
            map_p = cv.polylines(self.zmd.map_planning, [pp], isClosed=True, color=(100, 100, 100), thickness=2)
            map_p_fill = self.zmd.map_planning.copy()
            cv.fillPoly(map_p_fill, [pp], color=self.colors[i])
            alpha = 0.1
            self.zmd.map_planning = cv.addWeighted(map_p_fill, alpha, map_p, 1 - alpha, 0)
            # add cells to map_segmentation
            map_s = cv.polylines(self.zmd.map_segmentation, [pp], isClosed=True, color=(100, 100, 100), thickness=2)
            map_s_fill = self.zmd.map_segmentation.copy()
            cv.fillPoly(map_s_fill, [pp], color=self.colors[i])
            alpha = 0.1
            self.zmd.map_segmentation = cv.addWeighted(map_s_fill, alpha, map_s, 1 - alpha, 0)

            # add rectangle if rect_like
            if cell.rect_like:
                rp = cell.rectangle_poly.exterior.coords.xy
                rpoints = np.array(list(zip(rp[0], rp[1])), dtype=np.int32)
                self.zmd.map_planning = cv.polylines(self.zmd.map_planning, [rpoints], isClosed=True, color=(0, 0, 0), thickness=2)
                self.zmd.map_segmentation = cv.polylines(self.zmd.map_segmentation, [rpoints], isClosed=True, color=(0, 0, 0), thickness=2)

    def add_cell_paths_to_map_planning(self) -> None:
        map_planning = self.zmd.map_planning.copy()
        robot_r_px = int(self.pcfg.cleaning_r_ov / self.resolution)
        robot_d_px = robot_r_px * 2
        p_r = 5  # int(robot_r_px/10)
        for i, path_px in enumerate(self.paths_px):
            if path_px:
                pp = np.array(path_px.xy, dtype=np.int32)  # Ensure pp is a numpy array of integers
                map_planning = cv.polylines(map_planning, [pp], isClosed=False, color=self.colors[i], thickness=2)
                # plot cleaning
                overlay = map_planning.copy()
                cv.polylines(overlay, [pp], isClosed=False, color=self.colors[i], thickness=robot_d_px)
                alpha = 0.2  # Adjust transparency (0.0 transparent, 1.0 opaque)
                map_planning = cv.addWeighted(overlay, alpha, map_planning, 1 - alpha, 0)
                # add points
                for point in pp:
                    cv.circle(map_planning, tuple(point), p_r, self.colors[i], -1)
        self.zmd.map_planning = map_planning

    # filters ==========================================================================================================

    def apply_distance_filter_all_paths(self) -> None:
        """ Filter all paths based on the minimum distance between waypoints. """
        min_wp_dist_px = self.min_points_distance / self.resolution  # convert to px

        # pixel paths
        new_paths_px = []
        for path_px in self.paths_px:
            new_path_px = self.apply_distance_filter_path(path_px, min_wp_dist_px)
            new_paths_px.append(new_path_px)
        self.paths_px = new_paths_px

        # connection paths pixel
        new_c_paths_px = []
        for c_path_px in self.c_paths_px:
            new_c_path_px = self.apply_distance_filter_path(c_path_px, min_wp_dist_px)
            new_c_paths_px.append(new_c_path_px)
        self.c_paths_px = new_c_paths_px

        # meter paths
        self.paths_m = [self.path_px_to_m(path_px) for path_px in self.paths_px]

        # connection paths meter
        self.c_paths_m = [self.path_px_to_m(c_path_px) for c_path_px in self.c_paths_px]

    def apply_distance_filter_path(self, path: MyPath, min_wp_dist_px: int) -> MyPath:
        """ Filter the full path based on the minimum distance between waypoints. """
        new_points = []
        prev_point = path.point(0)
        new_points.append(prev_point)
        for i in range(1, path.n):
            point = path.point(i)
            dist = np.sqrt((point.x - prev_point.x) ** 2 + (point.y - prev_point.y) ** 2)
            if dist >= min_wp_dist_px:
                new_points.append(point)
                prev_point = point
        return MyPath(new_points)

    def process_full_path_distances(self) -> None:
        """ Process the full path distances based on the minimum distance between waypoints.
            If distance between points is less than min_points_distance, remove the point.
            If distance between points is bigger than min_points_distance, add as many points as needed.
        """

        if self.f_path_px is None:
            self.lg.error("Full path is empty or has less than 2 points. Cannot process distances.")
            return

        # if distance between points is less than min_points_distance, remove the point
        min_wp_dist_px = self.min_points_distance / self.resolution  # convert to px
        new_points = []
        prev_point = self.f_path_px.point(0)
        new_points.append(prev_point)
        for i in range(1, self.f_path_px.n):
            point = self.f_path_px.point(i)
            dist = np.sqrt((point.x - prev_point.x) ** 2 + (point.y - prev_point.y) ** 2)
            if dist >= min_wp_dist_px:
                new_points.append(point)
                prev_point = point
        self.f_path_px = MyPath(new_points)

        # if distance between points is bigger than min_points_distance, add as many points as needed
        new_points = []
        prev_point = self.f_path_px.point(0)
        new_points.append(prev_point)
        for i in range(1, self.f_path_px.n):
            point = self.f_path_px.point(i)
            dx = point.x - prev_point.x
            dy = point.y - prev_point.y
            dist = np.sqrt((dx) ** 2 + (dy) ** 2)
            if dist > 1.1*min_wp_dist_px:
                n_points = int(np.ceil(dist / min_wp_dist_px))
                for j in range(1, n_points):
                    new_x = prev_point.x + dx * j / n_points
                    new_y = prev_point.y + dy * j / n_points
                    new_points.append(MyPoint(new_x, new_y))
            new_points.append(point)
            prev_point = point
        self.f_path_px = MyPath(new_points)

        # create f_path_m
        self.f_path_m = self.path_px_to_m(self.f_path_px)

    # rotations and conversions ========================================================================================

    def path_m_to_px(self, path_m: MyPath) -> MyPath:
        return MyPath([self.point_m_to_px(p) for p in path_m.points])

    def path_px_to_m(self, path_px: MyPath) -> MyPath:
        return MyPath([self.point_px_to_m(p) for p in path_px.points])

    def point_m_to_px(self, p: MyPoint) -> MyPoint:
        return MyPoint(int(p.x / self.resolution), int(p.y / self.resolution))

    def point_px_to_m(self, p: MyPoint) -> MyPoint:
        return MyPoint(p.x * self.resolution, p.y * self.resolution)

    # Plots ============================================================================================================

    def plot_paths_px(self) -> None:
        """ Plot the calculated coverage paths (different colors) and connections between them. """
        if not self.dcfg.save_map_planning:
            return
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        self.plot_coverage_paths(ax)
        self.plot_connection_paths(ax)
        self.save_path_fig(fig)

    def plot_coverage_paths(self, ax):
        n = self.paths.n_paths
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
            ax.plot(pp[0, 0], pp[0, 1], 'o', color=colors[i], markersize=6)
            ax.text(pp[0, 0], pp[0, 1], f'S{i}', fontsize=9, color='k')
            ax.plot(pp[-1, 0], pp[-1, 1], 's', color=colors[i], markersize=6)
            ax.text(pp[-1, 0], pp[-1, 1], f'E{i}', fontsize=9, color='k')

    def plot_connection_paths(self, ax):
        for c_path in self.c_paths_px:
            if len(c_path.xy) < 2:
                continue
            pp = np.array(c_path.xy, dtype=np.int32)
            ax.plot(pp[0, 0], pp[0, 1], "o", color='gray', markersize=3)
            ax.plot(pp[-1, 0], pp[-1, 1], "o", color='gray', markersize=3)
            ax.plot(pp[:, 0], pp[:, 1], color='gray', linestyle=':')

    def save_path_fig(self, fig, is_anim=False):
        if not is_anim:
            fig_path = os.path.join(self.zmd.result_dir, "paths.png")
            fig.savefig(fig_path, format="png", dpi=self.dcfg.dpi)
        else:
            self.lg.info(f"saving animation ... n_frames: {self.f_path_px.n}")
            fig_path = os.path.join(self.zmd.result_dir, "anim_paths.gif")
            self.anim.save(fig_path)

    def plot_paths_m(self):
        _, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("paths_m")
        #
        n = self.st.n_paths
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
        # plt.show()

    def plot_final_path_m(self):
        """ Plot the final complete path in meters. """
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Final Path")
        scale = min(self.zmd.hw)*self.zmd.resolution/10
        if self.f_path_m is not None:
            pp = np.array(self.f_path_m.xy, dtype=np.float32)
            ax.plot(pp[:, 0], pp[:, 1], "o", color='blue', linewidth=2/scale, linestyle='-', markersize=3/scale)
        fig_path = os.path.join(self.zmd.result_dir, "final_path")
        # fig.savefig(fig_path+'.png', format="png", dpi=self.dcfg.dpi)
        fig.savefig(fig_path+'.svg', format="svg", dpi=self.dcfg.dpi)
        # plt.show()

    # Animate path =====================================================================================================

    def animate_robot_paths(self):
        # colors = get_n_colors_plt(n)
        fig, ax = plt.subplots()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Coverage Path Planning Animation")

        # plot polygons
        x, y = self.map_polygons.b_poly.exterior.xy
        ax.plot(x, y, color="black", linewidth=1)
        for obs in self.map_polygons.obs_polys:
            x, y = obs.exterior.xy
            ax.plot(x, y, color="black", linewidth=1)

        # # linewidth
        # mx = self.zmd.orig_map_data.updated_w * self.zmd.resolution  # original map size in m
        # nx = int(fig.get_figwidth() * fig.dpi)
        # lw = 0.5 * self.cleaning_width * nx / mx
        # self.lg.info("lw:", lw)

        # paths
        line_0, = ax.plot(self.f_path_px.xy[:2, 0], self.f_path_px.xy[:2, 1], "-", color='gray', linewidth=20, alpha=0.5)
        line_1, = ax.plot(self.f_path_px.xy[:2, 0], self.f_path_px.xy[:2, 1], "o-", color='green', markersize=5, linewidth=2)
        self.anim_line = [line_0, line_1]
        self.animate(fig)
        self.save_path_fig(fig, is_anim=True)
        # plt.show()

    def animate(self, fig):
        self.anim = animation.FuncAnimation(fig, self.anim_update, init_func=self.init_anim,
                                            frames=self.f_path_px.n, repeat=False, interval=1, blit=True)

    def init_anim(self):
        self.anim_line[0].set_data([0], [0])
        self.anim_line[1].set_data([0], [0])
        return self.anim_line

    def anim_update(self, i):
        if i >= self.f_path_px.n:
            return self.anim_line
        xx = self.f_path_px.xy[:i, 0]
        yy = self.f_path_px.xy[:i, 1]
        self.anim_line[0].set_data(xx, yy)
        self.anim_line[1].set_data(xx, yy)
        return self.anim_line

    # def rotate_waypoints(self) -> None:
    #     rect_data = self.zmd.final_rect_data
    #     rot = rect_data["rot"]
    #     tl = self.zmd.final_rectangle.tl
    #     resol = self.zmd.resolution
    #     x_0, y_0 = tl[0]*resol, tl[1]*resol  # first vertex (from which rot is calculated)
    #     # rotate
    #     new_paths = [[self.rotate_point(p, rot) for p in path_m.points] for path_m in self.paths_m]
    #     # translate
    #     new_paths = [MyPath([MyPoint(p.x + x_0, p.y + y_0) for p in new_path]) for new_path in new_paths]
    #     # update
    #     self.paths_m = new_paths

    # def rotate_point(self, p: MyPoint, rotation) -> MyPoint:
    #     x_r = p.x * np.cos(rotation) - p.y * np.sin(rotation)
    #     y_r = p.x * np.sin(rotation) + p.y * np.cos(rotation)
    #     return MyPoint(x_r, y_r)
