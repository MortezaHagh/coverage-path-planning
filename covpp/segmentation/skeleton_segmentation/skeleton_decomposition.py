"""Skeleton-based segmentation for optimal workspace decomposition."""
import cv2 as cv
import numpy as np
from networkx import Graph
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage.morphology import medial_axis  # , skeletonize
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter1d
from covpp.base.logger import MyLogger
from covpp.base.plot_styles import set_plot_styles, configure_ax
from covpp.base.configs import RobotConfig, PipeLineConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.skeleton_segmentation.skelet_branch_graph import SkeletBranchGraph as SBG
from covpp.segmentation.skeleton_segmentation.segmented_solution import CreateSegmentationSolution
from covpp.segmentation.skeleton_segmentation.data_structures import Skelet, Branch, CriticalPoint
from covpp.segmentation.skeleton_segmentation.critical_points_operations import CriticalPointOperations

# set np seed
np.random.seed(42)

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
        # n = max(2 * self.robot_radius_px // 4 + 1, 3)
        # structure = np.ones((n, n), dtype=np.uint8)
        map_ws = binary_closing(map_ws, iterations=3).astype(np.uint8) * 255
        map_ws = binary_opening(map_ws, iterations=3).astype(np.uint8) * 255
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

        # update skeleton_map_b based on branches
        skeleton_map_b = np.zeros_like(self.skelet.skeleton_map_b, dtype=bool)
        for branch in branches:
            for y, x in branch.points:
                skeleton_map_b[y, x] = True
        self.skelet.skeleton_map_b = skeleton_map_b

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

        critical_pts = []
        ep_thresh = 2 * self.robot_radius_px
        space_thresh = 4 * self.robot_radius_px

        # from scipy.signal import savgol_filter, find_peaks
        # mins, props = find_peaks(-smooth, prominence=1, width=self.robot_radius_px * 2, distance=self.robot_radius_px * 2)
        # from sklearn.tree import DecisionTreeRegressor

        print("robot radius px:", self.robot_radius_px)

        #
        sigma = max(int(self.robot_radius_px * 2.0), 5)
        # #
        # window_length = self.robot_radius_px * 1.0
        # window_length = max(21, 2 * (sigma // 2) + 1)  # window length must be odd and greater than polyorder

        for i, branch in enumerate(self.branches):

            n = len(branch.points)
            if n < (4 * self.robot_radius_px + 1):
                self.lg.error(f"Branch {i} is too short ({n} points). Skipping critical point detection.")
                continue

            if 1730 < n < 1750:
                pass

            dists = self.skelet.distance_map[branch.points[:, 0], branch.points[:, 1]].astype(float)
            dist_in = dists
            sigma_ = sigma

            # n_ds = int(n // (self.robot_radius_px))
            # idxs_ds = np.linspace(0, n - 1, num=n_ds, dtype=int)
            # dists_ds = dists[idxs_ds]  # downsample for smoothing
            # dist_in = dists_ds
            # sigma_ = int(sigma / n_ds) + 1

            # filter
            f_d = gaussian_filter1d(dist_in, sigma=sigma_, radius=sigma_ * 2)
            f_d = gaussian_filter1d(f_d, sigma=sigma_)
            # f_d_sv_p2 = savgol_filter(dists, window_length=window_length, polyorder=2)

            # # filter even wide segments with small change in distance to obstacles using decision tree regressor
            # model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=max(2 * self.robot_radius_px, 5))
            # model.fit(np.arange(len(f_d_gaus_2)).reshape(-1, 1), f_d_gaus_2)
            # d_pca = model.predict(np.arange(len(f_d_gaus_2)).reshape(-1, 1))

            smooth = f_d
            smooth = self.process_dists(smooth)
            smooth_nor = smooth / np.max(smooth)
            grad = np.gradient(smooth_nor)
            curv = np.gradient(grad)

            # grads_tr = grad[ep_thresh:-ep_thresh]
            # if all(grads_tr > 0) or all(grads_tr < 0):
            #     continue

            cpts = self.find_corridors(branch.points, smooth, space_thresh)
            # cpts = self.find_branch_critical_points_rel_grad(branch.points, f_d_gaus, space_thresh)
            # cpts = self.find_branch_critical_points_changepoint(branch.points, f_d_gaus)
            # cpts = self.find_critical_points_gradient(branch.points, f_d_gaus)
            # cpts = self.find_min_dist_cri00000.tical_points_global(branch.points, smooth)
            # cpts = self.find_min_dist_critical_points_local(branch.points, f_d_gaus)
            critical_pts.extend(cpts)

            if 600 < n < 606:
                _, axs = plt.subplots(2, 1, constrained_layout=True)
                ax1 = axs[0]
                ax1.plot(dists, label='Original')
                # ax1.plot(idxs_ds, dists_ds, label='Downsampled', linewidth=2.5)
                ax1.plot(smooth, label='Smoothed', linewidth=2.5)
                #
                if len(cpts) > 0:
                    cpt_indices = [np.where((branch.points == cp.coord).all(axis=1))[0][0] for cp in cpts]
                    ax1.scatter(cpt_indices, f_d[cpt_indices], color='red', label='Critical Points')
                ax1.set_title(f'Distance to Obstacles along Branch {i}')
                ax1.set_xlabel('Point Index along Branch')
                ax1.set_ylabel('Distance to Obstacles (pixels)')
                ax1.legend()
                ax1.grid()
                #
                # plot gradient and curvature
                ax2 = axs[1]
                ax2.plot(grad, label='Gradient', color='orange')
                ax2.plot(curv, label='Curvature', color='purple')
                ax2.set_title(f'Gradient and Curvature along Branch {i}')
                ax2.set_xlabel('Point Index along Branch')
                ax2.set_ylabel('Value')
                ax2.legend()
                ax2.grid()
                plt.show()

        return critical_pts

    def process_dists(self, dists: np.ndarray) -> np.ndarray:
        n = len(dists)
        thresh = max(2, n / 100)
        space_thresh = 2

        start_d = dists[0]
        start = 0
        i = 0
        while i < n - 1:
            if abs(dists[i] - start_d) > thresh:
                end = i
                if (end - start) > space_thresh:
                    dists[start:end] = np.mean(dists[start:end])
                start_d = dists[i]
                start = i
            i += 1
        if (i - start) > space_thresh:
            dists[start:i] = np.mean(dists[start:i])
        return dists

    def find_corridors(self, bpts: list[tuple], dists: np.ndarray, space_thresh: int):

        n = len(dists)
        d_min = np.min(dists)
        if d_min > (0.2 * n):
            return []

        min_cor_len = int(max(8 * self.robot_radius_px, n * 0.45))
        thresh = max(2, n / 200)

        cors = []
        start_d = dists[1]
        start = 1
        i = 1
        while i < n - 1:
            if abs(dists[i] - start_d) > thresh:
                end = i
                if end - start + 1 >= min_cor_len:
                    start_d = np.mean(dists[start:end])

                    cors.append((start, end))
                start_d = dists[i]
                start = i
            i += 1
        if i - 1 - start >= min_cor_len:
            cors.append((start, i - 1))

        cps = []
        for cor in cors:
            start, end = cor
            cps.append(CriticalPoint(coord=bpts[start], type='corridor'))
            cps.append(CriticalPoint(coord=bpts[end], type='corridor'))
        return cps

    # def find_corridors(self, bpts: list[tuple], dists: np.ndarray, space_thresh: int):

    #     n = len(dists)

    #     d_min = np.min(dists)
    #     d_max = np.max(dists)
    #     dl = max(0.1 * (d_max - d_min), 1)
    #     d_corr = d_min + dl

    #     if d_min > (0.3 * n):
    #         return []

    #     min_cor_len = int(max(8 * self.robot_radius_px, n * 0.6))

    #     cors = []
    #     start = None
    #     end = None
    #     i = 1
    #     while i < n - 1:
    #         if dists[i] < d_corr:  # threshold for corridor detection # param
    #             if start is None:
    #                 start = i
    #         else:
    #             if start is not None:
    #                 end = i
    #                 if end - start + 1 >= min_cor_len:
    #                     cors.append((start, end))
    #                 start = None
    #         i += 1
    #     if start is not None and end is None:
    #         end = n - 2
    #         if end - start + 1 >= min_cor_len:
    #             cors.append((start, end))

    #     cps = []
    #     for cor in cors:
    #         start, end = cor
    #         cps.append(CriticalPoint(coord=bpts[start], type='corridor'))
    #         cps.append(CriticalPoint(coord=bpts[end], type='corridor'))
    #     return cps

    # def find_corridors(self, bpts: list[tuple], dists: np.ndarray, space_thresh: int):

    #     dists = dists / np.max(dists)  # normalize distances to [0, 1]
    #     grads = np.gradient(dists)

    #     n = len(grads)
    #     min_cor_len = max(8 * self.robot_radius_px, n // 2)

    #     cors = []
    #     start = None
    #     end = None
    #     i = 0
    #     while i < n:
    #         if abs(grads[i]) < 0.002:  # threshold for corridor detection # param
    #             if start is None:
    #                 start = i
    #         else:
    #             if start is not None:
    #                 end = i
    #                 if end - start + 1 >= min_cor_len:
    #                     cors.append((start, end))
    #                 start = None
    #         i += 1
    #     if start is not None and end is None:
    #         end = n - 1
    #         if end - start + 1 >= min_cor_len:
    #             cors.append((start, end))

    #     cps = []
    #     for cor in cors:
    #         start, end = cor
    #         cps.append(CriticalPoint(coord=bpts[start], type='corridor'))
    #         cps.append(CriticalPoint(coord=bpts[end], type='corridor'))
    #     return cps

    def find_branch_critical_points_rel_grad(self, bpts: list[tuple], dists: np.ndarray, space_thresh: int) -> list[CriticalPoint]:

        # dowmn sample the distance array to reduce noise and computation
        n = len(dists)
        n1 = int(n // (self.robot_radius_px / 2))
        idxs_ds = np.linspace(0, n - 1, num=n1, dtype=int)
        dists_ds = dists[idxs_ds]

        # Parameters for corridor width change detection
        width_change_thresh = 0.1  # relative width change threshold

        # Calculate relative width changes
        rel_width_change = np.gradient(dists_ds) / dists_ds

        # Identify candidate critical points based on width change threshold
        candid_idx_ds = np.where(np.abs(rel_width_change) > width_change_thresh)[0]

        if len(candid_idx_ds) == 0:
            return cpts

        candid_idxs_real = idxs_ds[candid_idx_ds]

        # Filter candidates based on minimum spacing
        keep_indices = []
        for idx in candid_idxs_real:
            if len(keep_indices) == 0 or idx - keep_indices[-1] > space_thresh:
                keep_indices.append(idx)

        # Create CriticalPoint objects for the filtered indices
        for idx in keep_indices:
            cpts.append(CriticalPoint(coord=bpts[idx]))

        return cpts

    def find_branch_critical_points_changepoint(self, bpts: list[tuple], dists: np.ndarray) -> list[CriticalPoint]:
        import ruptures as rpt

        cpts = []
        if len(dists) < 12:
            return cpts

        dists = dists.reshape(-1, 1)
        min_size = max(5, self.robot_radius_px * 1)
        jump = max(1, self.robot_radius_px * 1)

        model = "rbf"  # "rbf"  # "linear"
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(dists)
        bkps = algo.predict(pen=self.robot_radius_px * 4)

        new_cps = []
        for idx in bkps[:-1]:

            if idx < self.robot_radius_px:
                continue

            if idx > len(dists) - self.robot_radius_px:
                continue

            new_cps.append(idx)
            cpts.append(CriticalPoint(coord=bpts[idx]))

        return cpts

    def find_critical_points_gradient(self, bpts: list[tuple], dists: np.ndarray) -> list[CriticalPoint]:
        """Detect significant radius transitions along each branch."""

        cpts = []

        min_spacing = 2 * self.robot_radius_px
        grad_thresh = 0.25      # relative radius change, bigger than this threshold is considered significant
        curv_thresh = 0.06      # recommended

        if len(dists) < 6:
            return cpts

        grad = np.gradient(dists)
        curv = np.gradient(grad)
        mean_radius = np.mean(dists)
        candidate = np.where(np.abs(grad) > grad_thresh)[0]

        if len(candidate) == 0:
            return cpts

        keep = []
        for idx in candidate:

            if idx < self.robot_radius_px:
                continue
            if idx > len(dists) - self.robot_radius_px:
                continue

            if len(keep) == 0:
                keep.append(idx)
                continue

            if idx - keep[-1] > min_spacing:
                keep.append(idx)

        for idx in keep:
            cpts.append(CriticalPoint(coord=bpts[idx]))

        return cpts

    def find_min_dist_critical_points_global(self, bpts: list[tuple], dists: np.ndarray) -> list[CriticalPoint]:
        """Find critical points based on local minimum distance to obstacles along branches."""
        cr_pts = []
        robot_radius_px = self.robot_radius_px
        search_r_px = 4 * robot_radius_px  # param # search radius for local minimum distance
        # search_r_px_2 = search_r_px // 2

        mean_dist = np.mean(dists)
        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]

        # skip endpoints and branchpoints
        if min_dist_idx in [0, len(dists) - 1]:
            return cr_pts

        # check if the minimum distance is significantly less than the mean distance
        p_significant = 0.9  # param
        if min_dist > p_significant * mean_dist:
            return cr_pts

        cp = CriticalPoint(coord=bpts[min_dist_idx])
        cr_pts.append(cp)
        return cr_pts

    def find_min_dist_critical_points_local(self, bpts: list[tuple], dists: np.ndarray) -> list[CriticalPoint]:
        """Find critical points based on local minimum distance to obstacles along branches."""
        cr_pts = []
        search_r = 4 * self.robot_radius_px  # param # search radius for local minimum distance
        search_d = search_r * 2

        n = len(bpts)
        i_skip = self.robot_radius_px

        if n < 2 * i_skip + 1:
            return cr_pts
        if n < search_r:
            search_r = n // 2

        dists = self.skelet.distance_map[np.array([p[0] for p in bpts]), np.array([p[1] for p in bpts])]
        visited = np.zeros(n, dtype=bool)

        for i, dist in enumerate(dists):
            if visited[i]:
                continue
            if i < i_skip or i >= n - i_skip:
                continue
            local_window = dists[max(0, i - search_r):min(n, i + search_r + 1)]

            mean_dist_local = np.mean(local_window)
            min_dist_local = np.min(local_window)

            if min_dist_local > 0.96 * mean_dist_local:  # param
                continue

            if dist == np.min(local_window):
                cp = CriticalPoint(coord=bpts[i], type='local_min')
                cr_pts.append(cp)
                visited[i:min(n, i + search_r + 1)] = True
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
        ax.contour(self.map_ws, levels=[0.5], colors='black', linewidths=2 / scale_px)
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
        ax2.contour(self.map_ws, levels=[0.5], colors='black', linewidths=4 / scale_px, zorder=0)
        # branches
        for i, branch in enumerate(self.branches):
            label = "Branch" if i == 0 else None
            yy, xx = branch.points[:, 0], branch.points[:, 1]
            color = 'orange' if branch.end_type == 'skip' else colors(i)
            # Draw branch lines below points
            ax2.plot(xx, yy, color=color, linewidth=4 / scale_px, linestyle='-', label=label, zorder=2)
            ax2.text(xx[len(xx) // 2], yy[len(yy) // 2], f'B{i}/{len(branch.points)}', color='black', fontsize=8, ha='center', va='center', zorder=3)

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
