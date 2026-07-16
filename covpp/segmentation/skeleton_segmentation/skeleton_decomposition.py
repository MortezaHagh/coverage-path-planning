"""Skeleton-based segmentation for optimal workspace decomposition."""
import cv2 as cv
import numpy as np
from networkx import Graph
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage.morphology import medial_axis  # , skeletonize
from scipy.ndimage import binary_opening, binary_closing
from covpp.base.logger import MyLogger
from covpp.base.plot_styles import set_plot_styles, configure_ax
from covpp.base.configs import RobotConfig, PipeLineConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.skeleton_segmentation.skelet_branch_graph import SkeletBranchGraph as SBG
from covpp.segmentation.skeleton_segmentation.segmented_solution import CreateSegmentationSolution
from covpp.segmentation.skeleton_segmentation.data_structures import Skelet, Branch, CriticalPoint
from covpp.segmentation.skeleton_segmentation.critical_points_operations import CriticalPointOperations

OCCUPIED = 0
FREE = 255


class SkeletonDecomposition:
    """Skeletonization-based decomposition of the zone map.

    Steps:
    - Create skeleton of the workspace.
    - Extract skeleton branches between end points and branch points.
    - Find critical points in branches based on local minimum distance to obstacles.
    - For each critical point, find two base points by expanding side points until hitting contours.
    - Create segmented map by connecting base points and filling areas.
    - Create polygons for each segment.
    - Create adjacency matrix based on critical points connecting segments.
    """

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("SkeletonDecomposition")
        self.lg.info2("Initializing Skeleton Decomposition.")

        # data
        self.zmd = zmd
        self.robot_radius_px = int(RobotConfig.radius / zmd.resolution)
        self.map_ws: np.ndarray = None   # uint8 binary map (OCCUPIED, FREE)

        # attributes
        self.skelet: Skelet = None                          # skeletonization data
        self.skeleton_graph: Graph = None                   # skeleton graph
        self.branches: list[Branch] = []                    # skeleton branches
        self.bpts_graph: Graph = None                       # graph of branch points
        self.critical_pts: list[CriticalPoint] = []         # critical points data
        self.base_pts: list[list[tuple, tuple]] = []       # base points for critical points (n_cp * 2)
        self.branch_cr_pts: list[list[CriticalPoint]] = []  # critical points for each branch
        self.local_min_cr_pts: list[CriticalPoint] = []     # additional critical points based on local minimum distance to obstacles
        self.erp_sidep = []                                 # for debugging
        self.error_point_segments = []                      # for debugging

        # results
        self.polygons: list[Polygon] = []                   # shapely polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray      # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray      # map_ws with segmentation lines drawn

    def run(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        # create skeleton and extract special points and branches
        self.create_skeleton()
        self.create_branches()

        # find critical points and base points
        critical_operator = CriticalPointOperations(self.skelet.distance_map, self.skelet.skeleton_map_b, self.branches)
        # critical_operator.run()
        # self.critical_pts = critical_operator.critical_pts
        # self.base_pts = critical_operator.base_pts
        # self.branch_cr_pts = self.critical_pts.copy()

        # additional critical points and base points based on local minimum distance to obstacles
        new_cr_pts = self.find_min_dist_critical_points()
        new_base_pts = critical_operator.find_base_points(critical_pts=new_cr_pts)
        self.local_min_cr_pts = new_cr_pts
        self.critical_pts += new_cr_pts
        self.base_pts += new_base_pts

        # create segmented solution
        css = CreateSegmentationSolution(map_ws=self.zmd.map_ws.copy(),
                                         map_ws_segmented=self.zmd.map_ws_segmented.copy(),
                                         robot_radius_px=self.robot_radius_px,
                                         origin_coord=self.zmd.global_xy)
        css.run_by_critical_points(base_pts=self.base_pts, critical_pts=self.critical_pts)
        self.polygons = css.polygons
        self.map_ws_segmented = css.map_ws_segmented
        self.adjacency_matrix = css.adjacency_matrix
        self.error_point_segments = css.error_point_segments

        self.lg.debug(f"Extracted {len(self.polygons)} polygons.")
        self.plot_segmentation()

    # ==================================================================================================================
    # skeletonization and special points extraction
    # ==================================================================================================================

    def create_skeleton(self) -> None:
        """Create Skelet object. """

        map_ws = self.zmd.map_ws.copy()
        # self.map_ws = map_ws.astype(np.uint8) * 255
        map_ws = binary_closing(map_ws, iterations=2).astype(np.uint8) * 255
        map_ws = binary_opening(map_ws, iterations=2).astype(np.uint8) * 255
        self.map_ws = map_ws

        # Compute medial axis (skeleton) and the distance transform
        medial_axis_map, distance_map = medial_axis(map_ws, return_distance=True)
        skeleton_map_b = medial_axis_map

        # distance transform and normalization
        distance_norm_map = (distance_map / distance_map.max() * FREE).astype(np.uint8)  # dtype=uint8

        # # skeletonization
        # skeleton_map_b = skeletonize(map_ws, method='lee')

        # distance_normalized masked by skeleton
        skeleton_distance_map = distance_norm_map * skeleton_map_b

        # # debugging
        # self.viz_img(skeleton_map_b.astype(np.uint8) * 255, "skeleton_map_b")

        # ----------------------------------------------------------------------
        # Skeleton Segmented Map Creation
        skeleton_segmented_map = map_ws.copy()                      # start with workspace map
        skeleton_segmented_map[skeleton_map_b] = OCCUPIED           # set skeleton pixels to OCCUPIED
        skeleton_segmented_map = FREE - skeleton_segmented_map      # invert map, so skeleton contours are closed areas
        # find contours and fill areas
        contours2, _ = cv.findContours(skeleton_segmented_map, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        # fill area with different gray value
        if len(contours2) > 254:
            self.lg.warn(f"Too many contours found: {len(contours2)}.")
            skeleton_segmented_map = skeleton_segmented_map.astype(np.uint16)   # use uint16 to support more contours
        for i, cnt in enumerate(contours2):
            cv.fillPoly(skeleton_segmented_map, pts=[cnt], color=i + 1)
        skeleton_segmented_map[skeleton_map_b] = OCCUPIED                       # make skeleton pixels OCCUPIED again

        # # debugging
        # self.viz_img(skeleton_segmented_map, "Skeleton Segmented Map")

        # final skeletonization data
        self.skelet = Skelet(
            skeleton_map_b=skeleton_map_b,
            distance_map=distance_norm_map,
            skeleton_segmented_map=skeleton_segmented_map,
            skeleton_distance_map=skeleton_distance_map,
            end_points=None,
            branch_pts=None,
            f_end_pts=None
        )

    def create_branches(self) -> None:
        """ Extract branches from a skeleton as paths between end_points/branch_pts. """

        # maps = SBG.create_skeletonized_maps(self.zmd.map_ws, method='medial_axis')
        sk_graph = SBG.create_skeleton_graph(self.skelet.skeleton_map_b)

        # update skeletonization data with special points
        end_pts, branch_pts = SBG.create_especial_points(sk_graph)
        self.skelet.end_points = np.array(end_pts)
        self.skelet.f_end_pts = np.array(end_pts)
        self.skelet.branch_pts = np.array(branch_pts)
        f_end_pts, skipped_end_pts = self.filter_end_points(end_pts)
        especial_pts_dict = {"end_pts": f_end_pts, "branch_pts": branch_pts, "skipped_end_pts": skipped_end_pts}

        min_dist_thresh_branch_px = 4 * self.robot_radius_px
        branches, bpts_graph = SBG.create_branch_points_graph(sk_graph=sk_graph,
                                                              especial_pts_dict=especial_pts_dict,
                                                              len_thresh_px=min_dist_thresh_branch_px)

        # update class attributes
        self.skeleton_graph = sk_graph
        self.branches = branches
        self.bpts_graph = bpts_graph

        # SBG.plot_branch_graph(branches, bpts_graph, self.map_ws)

        # validate
        if len(self.branches) == 0:
            self.lg.warn("No branches extracted from skeleton.")

    def filter_end_points(self, end_points: list[tuple]) -> tuple[list[tuple], list[tuple]]:
        """ Filter end points based on their distance_map values. Keep only those that have distance_map values greater than a threshold. """

        threshold = 2 * self.robot_radius_px
        final_end_pts = []
        skipped_end_pts = []
        for pt in end_points:
            dist = self.skelet.distance_map[pt[0], pt[1]]
            if dist < threshold:
                skipped_end_pts.append(pt)
            else:
                final_end_pts.append(pt)
        self.skelet.f_end_pts = np.array(final_end_pts)
        return final_end_pts, skipped_end_pts

    # ==================================================================================================================
    # Critical points based on local minimum distance to obstacles along branches
    # ==================================================================================================================

    def find_min_dist_critical_points(self) -> list[CriticalPoint]:
        """Find critical points based on local minimum distance to obstacles along branches."""
        cr_pts = []
        robot_radius_px = self.robot_radius_px
        search_r_px = 4 * robot_radius_px  # param # search radius for local minimum distance
        # search_r_px_2 = search_r_px // 2

        for branch in self.branches:
            r_thresh = search_r_px
            n = len(branch.points)
            if n < search_r_px:
                r_thresh = n // 2
            i_min = n // 10
            dists = self.skelet.distance_map[branch.points[:, 0], branch.points[:, 1]]
            # max_dist = np.max(dists)
            mean_dist = np.mean(dists)
            visited = np.zeros(n, dtype=bool)

            for i, dist in enumerate(dists):
                if visited[i]:
                    continue
                if i < i_min or i >= n - i_min:
                    continue
                local_window = dists[max(0, i - r_thresh):min(n, i + r_thresh + 1)]

                # mean_dist_local = np.mean(local_window)
                min_dist_local = np.min(local_window)

                # skip if minimum distance is not significantly less than mean distance
                if min_dist_local > 0.9 * mean_dist:  # param
                    continue

                if dist == np.min(local_window):
                    # found a local minimum distance point, create a CriticalPoint object
                    cp = CriticalPoint(coord=branch.points[i])
                    cr_pts.append(cp)
                    visited[max(0, i - r_thresh):min(n, i + r_thresh + 1)] = True  # mark as visited
        self.lg.debug(f"Found {len(cr_pts)} critical points based on local minimum distance.")
        return cr_pts

    # ==================================================================================================================
    # visualization methods
    # ==================================================================================================================

    def plot_segmentation(self) -> None:
        """Visualize skeletonization results."""

        # professional styles
        fig_w = 8.27  # A4 width in inches
        fig_h = fig_w * self.zmd.hw[0] / self.zmd.hw[1]
        set_plot_styles(fig_w, fig_h)

        # scale_m: float = min(zmd.hw)*zmd.resolution/10
        # A4 size: 8.27 x 11.69 inches
        scale_px = min(self.zmd.hw) / 500.0
        marker_size = 20 / scale_px
        colors = plt.cm.get_cmap('hsv', len(self.branches))

        # data
        end_pts = self.skelet.end_points
        branch_pts = self.skelet.branch_pts
        f_end_pts = self.skelet.f_end_pts

        # --------------------------------------------------------------------------------------------------------------
        # Figure 1: skeleton and endpoints/branchpoints
        fig1, ax = plt.subplots(1, 1, constrained_layout=True)

        # plot skeleton (medial) axis - endpoints and branchpoints in same plot
        ax.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=2 / scale_px)
        ax.imshow(FREE - self.skelet.skeleton_distance_map, cmap='bone', alpha=1.0)
        if end_pts.size > 0:
            ax.scatter(end_pts[:, 1], end_pts[:, 0], color='blue', s=marker_size, label='Endpoints')
        ax.scatter(branch_pts[:, 1], branch_pts[:, 0], color='red', s=marker_size, label='Branchpoints')
        if len(f_end_pts) > 0:
            ax.scatter(f_end_pts[:, 1], f_end_pts[:, 0], color='lime', s=marker_size, label='Filtered Endpoints')
        if len(self.local_min_cr_pts) > 0:
            local_min_cp_coords = np.array([cp.coord for cp in self.local_min_cr_pts])
            ax.scatter(local_min_cp_coords[:, 1], local_min_cp_coords[:, 0], color='orange', s=marker_size, label='Local Min Critical Points')
        if len(self.branch_cr_pts) > 0:
            branch_cr_pts_coords = np.array([cp.coord for cp in self.branch_cr_pts])
            ax.scatter(branch_cr_pts_coords[:, 1], branch_cr_pts_coords[:, 0], color='k', s=marker_size // 1.5, label='Branch Critical Points')
        # add a circle around the robot radius for reference
        xc, yc = self.zmd.hw[1] // 2, self.zmd.hw[0] // 2
        circle = plt.Circle((xc, yc), self.robot_radius_px, color='green', fill=False, linewidth=2 / scale_px)
        ax.add_artist(circle)
        configure_ax(ax, 'Skeleton with Endpoints and Branchpoints', show_title=False)

        # --------------------------------------------------------------------------------------------------------------
        # Figure 2: plot branches with critical points and base points
        fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)

        # Draw contours in the background
        ax2.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=4 / scale_px, zorder=0)
        # branches
        for i, branch in enumerate(self.branches):
            label = "Branch" if i == 0 else None
            yy, xx = branch.points[:, 0], branch.points[:, 1]
            color = 'orange' if branch.end_type == 'skip' else colors(i)
            # Draw branch lines below points
            ax2.plot(xx, yy, color=color, linewidth=4 / scale_px, linestyle='-', label=label, zorder=2)

        # plot error point if any
        if self.erp_sidep:  # no side points found,  len(side_pts) != 2:
            for iep, ep in enumerate(self.erp_sidep):
                ax2.scatter(ep[1], ep[0], color='red', s=marker_size * 3, zorder=1, label='erp_sidep' if iep == 0 else None)
        if self.error_point_segments:  # less than 2 segments found, not enough to create adjacency
            for iep, ep in enumerate(self.error_point_segments):
                ax2.scatter(ep[1], ep[0], color='cyan', s=marker_size * 3, zorder=1, label='error_point_segments' if iep == 0 else None)

        # for all critical points
        temp = True
        for cp in self.critical_pts:
            # critical point
            y, x = cp.coord
            ax2.scatter(x, y, color='red', alpha=1, s=marker_size, zorder=4, label='Critical Point' if temp else None)
            # base points
            for bp in cp.base_pts:
                ax2.scatter(bp[1], bp[0], color='blue', alpha=1, s=marker_size // 3, zorder=4, label='Base Point' if temp else None)
                temp = False
            # path to base points
            for path in cp.base_paths:
                path = np.array(path)
                ax2.plot(path[:, 1], path[:, 0], color='gray', linewidth=2 / scale_px, linestyle='-', zorder=2)

        # branch points
        ax2.scatter(branch_pts[:, 1], branch_pts[:, 0], color='k', s=marker_size // 2, label='Branchpoints')

        # finalize
        configure_ax(ax2, 'Detected Branches with Critical and Base Points', show_title=False, show_legend=True)
        ax2.invert_yaxis()

        # # --------------------------------------------------------------------------------------------------------------
        # save figures
        im_format = 'png'  # 'svg'
        if PipeLineConfig.do_plot_segmentation:
            save_dir = self.zmd.segmentation_method_dir
            fig1_path = save_dir + "/skeleton_end_branch_points" + f".{im_format}"
            fig2_path = save_dir + "/skeleton_branches_criticals" + f".{im_format}"
            fig1.savefig(fig1_path, format=im_format, dpi=PipeLineConfig.dpi, bbox_inches='tight')
            fig2.savefig(fig2_path, format=im_format, dpi=PipeLineConfig.dpi, bbox_inches='tight')

        plt.show()
        plt.close('all')

    @staticmethod
    def viz_img(img: np.ndarray, name: str):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
