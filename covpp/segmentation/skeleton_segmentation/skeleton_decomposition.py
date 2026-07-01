"""Skeleton-based segmentation for optimal workspace decomposition."""
import cv2 as cv
import numpy as np
from networkx import Graph
from matplotlib import pyplot as plt
from shapely.geometry import Polygon  # , LineString
from skimage.morphology import medial_axis  # , skeletonize
# from scipy.ndimage import binary_opening, binary_closing
#
from covpp.base.logger import MyLogger
from covpp.base.plot_styles import set_plot_styles
from covpp.base.configs import RobotConfig, PipeLineConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.skeleton_segmentation.skelet_branch_graph import SkeletBranchGraph as SBG
from covpp.segmentation.skeleton_segmentation.critical_points_operations import CriticalPointOperations
from covpp.segmentation.skeleton_segmentation.data_structures import Configs, Skelet, Branch, CriticalPoint, SegmentSolution

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

        # settings
        self.cfgs = Configs()

        # attributes
        self.skelet: Skelet = None                          # skeletonization data
        self.skeleton_graph: Graph = None                         # skeleton graph
        self.branches: list[Branch] = []                    # skeleton branches
        self.bpts_graph: Graph = None                       # graph of branch points
        self.critical_pts: list[CriticalPoint] = []         # critical points data
        self.base_pts: list[tuple[tuple, tuple]] = []       # base points for critical points (n_cp * 2)
        self.erp_sidep = []                                 # for debugging
        self.error_point_segments = []                      # for debugging

        # results
        self.polygons: list[Polygon] = []                   # shapely polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray      # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray      # map_ws with segmentation lines drawn

    def skeleton_decomposition(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        # create skeleton and extract special points and branches
        self.create_skeleton()
        self.create_branches()

        # find critical points and base points
        critical_operator = CriticalPointOperations(self.skelet.distance_map, self.skelet.skeleton_map_b, self.branches)
        critical_operator.run()
        self.critical_pts = critical_operator.critical_pts
        self.base_pts = critical_operator.base_pts

        # create segmented map
        segment_solution = self.create_segmented_map()

        # create adjacency matrix and polygons
        self.adjacency_matrix = self.create_adjacency_matrix(segment_solution)
        self.create_polygons(segment_solution.polygons)
        self.lg.info2(f"Extracted {len(self.polygons)} polygons.")

        self.plot_segmentation()

    # ==================================================================================================================
    # segmentation map, polygons and adjacency matrix creation
    # ==================================================================================================================

    def create_segmented_map(self) -> SegmentSolution:
        """ Create segmented map based on base points.

        Steps:
        - Connect each two base points with a line.
        - Find contours and fill areas to create segmented map.
        - Each area will have a unique gray value. (later used in creating cells)
        """

        seg_map = self.map_ws.copy()

        # create a line connecting each two base points consecutively
        for bases in self.base_pts:
            n_bases = len(bases)
            for i in range(n_bases):
                j = (i + 1) % n_bases
                base1 = bases[i]
                base2 = bases[j]
                cv.line(seg_map, (base1[1], base1[0]), (base2[1], base2[0]), OCCUPIED, thickness=2)

        # # # create map_ws_segmented
        self.map_ws_segmented = seg_map.copy()

        seg_ids = []

        # find contours and fill areas
        contours, hierarchy = cv.findContours(seg_map, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        self.lg.debug(f"Found {len(contours)} contours in segmented map.")

        # filter contours: remove those with children and small areas
        filtered_contours = []
        it = 0
        for c, h in zip(contours, hierarchy):

            # skip contours with children
            if not self.is_valid_contour(h, hierarchy):
                continue

            # minimum area threshold
            area = cv.contourArea(c)
            if area >= self.cfgs.min_area_thresh_for_contour:
                filtered_contours.append(c)
                it += 1

        # check too many contours
        self.lg.debug(f"Found {len(filtered_contours)} filtered contours in segmented map.")
        if len(filtered_contours) > 254:
            self.lg.warn(f"Too many contours found in segmented map: {len(filtered_contours)}.")

        # fill area with different gray value (used for segment IDs)
        for i, cnt in enumerate(filtered_contours):
            new_id = i + 1
            seg_ids.append(new_id)
            cv.fillPoly(seg_map, pts=[cnt], color=new_id)
            cv.drawContours(seg_map, [cnt], 0, FREE, thickness=1)

        # # debugging
        # self.viz_img(seg_map, "create_segmented_map")

        if not seg_ids:
            raise ValueError("No segments found in segmented map.")
        return SegmentSolution(seg_map, seg_ids, filtered_contours)

    def create_adjacency_matrix(self, segment_solution: SegmentSolution) -> np.ndarray:
        """ Create adjacency matrix between segments based on critical points.

        Method:
        - For each critical point, find the segments around it by checking a small canvas.
        - Mark those segments as adjacent in the adjacency matrix.
        """

        adjacency_matrix = np.zeros((len(segment_solution.seg_ids), len(segment_solution.seg_ids)), dtype=np.bool)
        h, w = segment_solution.segmented_map.shape
        # canvas_size = 40 // 2
        canvas_size = 2 * self.robot_radius_px  # canvas size based on robot radius

        for cp in self.critical_pts:
            y, x = cp.coord
            y_min = max(0, y - canvas_size)
            y_max = min(h, y + canvas_size)
            x_min = max(0, x - canvas_size)
            x_max = min(w, x + canvas_size)
            canvas = segment_solution.segmented_map[y_min:y_max, x_min:x_max]
            unique_ids = np.unique(canvas)
            # remove OCCUPIED and FREE
            unique_ids = unique_ids[(unique_ids != OCCUPIED) & (unique_ids != FREE)]
            n_ids = len(unique_ids)
            if n_ids < 2:
                self.error_point_segments.append(cp.coord)
                self.lg.warn(f"Less than 2 segments found near critical point at {cp.coord}.")

            for i in range(n_ids):
                for j in range(i + 1, n_ids):
                    adjacency_matrix[unique_ids[i] - 1, unique_ids[j] - 1] = True
                    adjacency_matrix[unique_ids[j] - 1, unique_ids[i] - 1] = True
        return adjacency_matrix

    def create_polygons(self, contours: list[np.ndarray]) -> None:

        # create shapely polygons for each contour
        from shapely.geometry import Polygon
        polygons = []
        for i, cnt in enumerate(contours):
            pts = cnt.reshape(-1, 2)
            if len(pts) < 3:
                self.lg.error(f"Contour {i} has less than 3 points, cannot create polygon.")
                continue
            # add global coordinate conversion
            g_pts = self.zmd.global_xy + pts
            poly = Polygon(g_pts)
            polygons.append(poly)
        self.polygons = polygons

    # ==================================================================================================================
    # skeletonization and special points extraction
    # ==================================================================================================================

    def create_skeleton(self) -> None:
        """Create Skelet object. """

        map_ws = self.zmd.map_ws.copy()
        self.map_ws = map_ws.astype(np.uint8) * 255
        # map_ws = binary_closing(map_ws, iterations=2).astype(np.uint8) * 255
        # map_ws = binary_opening(map_ws, iterations=2).astype(np.uint8) * 255
        # self.map_ws = map_ws

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

        # maps = SBG.create_maps(self.zmd.map_ws, method='medial_axis')
        sk_graph = SBG.create_skeleton_graph(self.skelet.skeleton_map_b)

        # update skeletonization data with special points
        end_pts, branch_pts = SBG.create_especial_points(sk_graph)
        self.skelet.end_points = np.array(end_pts)
        self.skelet.f_end_pts = np.array(end_pts)
        self.skelet.branch_pts = np.array(branch_pts)

        f_end_pts, skipped_end_pts = self.filter_end_points(end_pts)

        min_dist_thresh_branch_px = 4 * self.robot_radius_px
        especial_pts_dict = {"end_pts": f_end_pts, "branch_pts": branch_pts, "skipped_end_pts": skipped_end_pts}

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
    # Utility methods
    # ==================================================================================================================

    @staticmethod
    def find_level(h, hierarchy):
        """ Find the level of contour in hierarchy. """
        level = 0
        parent = h[3]
        while parent != -1:
            level += 1
            parent = hierarchy[parent][3]
        return level

    @staticmethod
    def is_valid_contour(h, hierarchy) -> bool:
        # if level is odd, it is a hole contour
        level = SkeletonDecomposition.find_level(h, hierarchy)
        if level % 2 == 1:
            return False
        return True
        # # Check if contour has both parent and children in hierarchy.
        # return not(h[3] != -1 and h[2] != -1)

    # ==================================================================================================================
    # visualization methods
    # ==================================================================================================================

    def plot_segmentation(self) -> None:
        """Visualize skeletonization results."""

        def configure_ax(ax, title, show_legend=True, show_title=True):
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            if show_title:
                ax.set_title(title, fontsize='medium')
            if show_legend:
                ax.legend(
                    frameon=True,
                    edgecolor='black',
                    fontsize=10,
                    loc='upper left',
                    bbox_to_anchor=(1.0, 1),
                    borderaxespad=0.0,
                )

        # professional styles
        fig_w = 8.27  # A4 width in inches
        fig_h = fig_w * self.zmd.hw[0] / self.zmd.hw[1]
        set_plot_styles(fig_w, fig_h)

        # scale_m: float = min(zmd.hw)*zmd.resolution/10
        # A4 size: 8.27 x 11.69 inches
        scale_px = min(self.zmd.hw) / 500.0
        marker_size = 50 / scale_px
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
        for cp in self.critical_pts:

            # critical point
            y, x = cp.coord
            ax2.scatter(x, y, color='red', alpha=1, s=marker_size, zorder=4, label='Critical Point' if cp == self.critical_pts[0] else None)

            # base points
            for bp in cp.base_pts:
                ax2.scatter(bp[1], bp[0], color='blue', alpha=1, s=marker_size, zorder=4, label='Base Point' if cp == self.critical_pts[0] else None)

            # path to base points
            for path in cp.base_paths:
                path = np.array(path)
                ax2.plot(path[:, 1], path[:, 0], color='gray', linewidth=4 / scale_px, linestyle='-', zorder=2)

        # finalize
        configure_ax(ax2, 'Detected Branches with Critical and Base Points', show_title=False, show_legend=True)
        ax2.invert_yaxis()

        # # --------------------------------------------------------------------------------------------------------------
        # # Figure 3: division map
        # _, ax3 = plt.subplots(1, 1, constrained_layout=True)
        # ax3.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=2/scale_px)
        # ax3.imshow(self.skelet.skeleton_segmented_map, cmap='tab20', alpha=1.0)
        # configure_ax(ax3, 'Division Map based on Skeleton', show_legend=False)

        # save figures
        im_format = 'png'  # 'svg'
        if PipeLineConfig.do_plot_segmentation:
            save_dir = self.zmd.segmentation_method_dir
            fig1_path = save_dir + "/skeleton_end_branch_points" + f".{im_format}"
            fig2_path = save_dir + "/skeleton_branches_criticals" + f".{im_format}"
            fig1.savefig(fig1_path, format=im_format, dpi=PipeLineConfig.dpi, bbox_inches='tight')
            fig2.savefig(fig2_path, format=im_format, dpi=PipeLineConfig.dpi, bbox_inches='tight')

        # plt.show()
        plt.close('all')

        # # plot division map separately
        # fig, ax = plt.subplots(1, 1, constrained_layout=True)
        # ax.imshow(self.skelet.skeleton_segmented_map, cmap='nipy_spectral', alpha=1.0)  # nipy_spectral tab20
        # ax.set_aspect('equal')
        # ax.axis('off')
        # fig3_path = save_dir + "/skeleton_division_map.pdf"
        # plt.show()
        # fig.savefig(fig3_path, format='pdf', dpi=600, bbox_inches='tight')

        # # cv.imshow("Division Map", self.skelet.skeleton_segmented_map)
        # # show critical points
        # viz_map = self.skelet.skeleton_segmented_map.copy()
        # viz_map[self.skelet.skeleton_map_b] = 0  # hide skeleton for better visualization
        # # ti uint8 for visualization
        # if viz_map.dtype != np.uint8:
        #     viz_map = viz_map.astype(np.uint8)
        # skel_div_map_cp = cv.cvtColor(viz_map, cv.COLOR_GRAY2BGR)
        # for cp in self.critical_pts:
        #     y, x = cp.coord
        #     skel_div_map_cp[y, x] = (0, 0, 255)
        # # show error points
        # for ep in self.erp_sidep:
        #     y, x = ep
        #     skel_div_map_cp[y, x] = (255, 0, 0)  # error points in blue
        # for ep in self.error_point_segments:
        #     y, x = ep
        #     skel_div_map_cp[y, x] = (255, 255, 0)  # error points in cyan
        # cv.imshow("Division Map with Critical Points", skel_div_map_cp)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    @staticmethod
    def viz_img(img: np.ndarray, name: str):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
