"""Skeleton-based segmentation for optimal workspace decomposition."""
import cv2 as cv
import numpy as np
from networkx import Graph
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage.morphology import medial_axis  # , skeletonize
from scipy.interpolate import splprep, splev
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


class SweepSkeletonDecomposition:
    """Skeletonization-based decomposition of the zone map.

    Steps:

    """

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("SweepSkeletonDecomposition")
        self.lg.info2("Initializing Sweep Skeleton Decomposition.")

        # data
        self.zmd = zmd
        self.robot_radius_px = int(RobotConfig.radius / zmd.resolution)
        self.map_ws: np.ndarray = None   # uint8 binary map (OCCUPIED, FREE)
        self.sweep_seg_map: np.ndarray = None  # uint8 map with sweep segmentation lines drawn

        # attributes
        self.critical_operator: CriticalPointOperations = None
        self.skelet: Skelet = None                          # skeletonization data
        self.skeleton_graph: Graph = None                   # skeleton graph
        self.branches: list[Branch] = []                    # skeleton branches
        self.bpts_graph: Graph = None                       # graph of branch points
        self.critical_pts: list[CriticalPoint] = []         # critical points data
        self.base_pts: list[list[tuple, tuple]] = []       # base points for critical points (n_cp * 2)

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
        self.critical_operator = critical_operator

        new_cr_pts = self.sweep_skeleton()
        new_base_pts = critical_operator.find_base_points(critical_pts=new_cr_pts)
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
        self.sweep_seg_map = map_ws.copy()

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
            distance_map=distance_map,
            distance_map_norm=distance_norm_map,
            skeleton_segmented_map=skeleton_segmented_map,
            skeleton_distance_map=skeleton_distance_map,
            end_points=None,
            branch_pts=None,
            f_end_pts=None
        )

    def create_branches(self) -> None:
        """ Extract branches from a skeleton as paths between end_points/branch_pts. """

        # maps = SBG.create_skeletonized_maps(self.zmd.map_ws, method='medial_axis')
        sk_graph = SBG.create_skeleton_graph(self.skelet.skeleton_map_b, self.skelet.distance_map)

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
    # Sweep Operation
    # ==================================================================================================================

    def sweep_skeleton(self) -> None:
        '''

        Steps:
        - Sort branches based on max_dists, for each branch:
        - traverse from start to end, compute the perpendicular obstacle distance at each point
        - find sudden changes in the distance profile, mark as critical points
        '''

        critical_pts = []
        p_dist_thresh = max(5, 2 * self.robot_radius_px)    # threshold for sudden change in perpendicular distance to obstacles
        # ds_step = max(2, self.robot_radius_px // 4)         # downsample step for branches
        ds_step = 2                                         # downsample step for branches
        l_init = max(4, self.robot_radius_px)               # initial and final length to ignore for each branch (to avoid endpoints)
        space_thresh = 4 * self.robot_radius_px             # threshold for minimum distance between critical points along the branch

        max_dists = [branch.max_dist for branch in self.branches]
        sorted_indices = np.argsort(max_dists)[::-1]  # from largest to smallest

        for i in sorted_indices:
            branch = self.branches[i]
            xys = branch.points

            n = len(xys)
            if n < 3 * l_init:
                continue
            xys = np.array(xys[l_init:n - l_init])

            # # # smooth the points using Gaussian filter
            # sigma = max(4, self.robot_radius_px // 2)
            # xys_s_1 = np.column_stack([
            #     gaussian_filter1d(xys[:, 0], sigma=sigma),
            #     gaussian_filter1d(xys[:, 1], sigma=sigma)])

            xys_s_2 = self.smooth_path(xys, smoothing_factor=2 * self.robot_radius_px)

            # downsample the points for faster processing
            ds_idxs = np.arange(0, len(xys_s_2), ds_step)

            # input points for perpendicular distance computation
            pts = [tuple((int(p[0]), int(p[1]))) for p in xys[ds_idxs]]

            # compute perpendicular distances to obstacles for each point in the branch
            cp_pts = []
            last_cp = None
            prev_d = None
            c_space_thresh = max(space_thresh, n // 20)
            # c_p_dist_thresh = max(p_dist_thresh, p_dist_min_ep / 2)
            for i, p1 in enumerate(pts[:-1]):
                if last_cp is not None:
                    if np.linalg.norm(np.array(p1) - np.array(last_cp)) < c_space_thresh:
                        continue
                p2 = pts[i + 1]
                p_d = self.find_max_obst_dist_point(p1, p2)
                if p_d is not None:
                    if prev_d is not None:
                        if abs(p_d - prev_d) > max(p_dist_thresh, max(prev_d, p_d) / 2):  # sudden change in distance (critical point)
                            cp_pts.append(pts[i])
                            last_cp = p1
                    prev_d = p_d

            # # find sudden changes in the perpendicular distance profile and mark as critical points
            # p_dist_min_ep = min(p_dists[0], p_dists[-1])
            # p_dists = np.array(p_dists)
            # c_p_dist_thresh = max(p_dist_thresh, p_dist_min_ep / 2)
            # grad = np.diff(p_dists)
            # cpis = np.where(np.abs(grad) > c_p_dist_thresh)
            # cpis = cpis[0]
            # if len(cpis) == 0:
            #     continue

            cpts = []
            if len(cp_pts) > 0:
                # cpts = [CriticalPoint(coord=pts[j], type='sweep') for j in cpis]
                cpts = [CriticalPoint(coord=p, type='sweep') for p in cp_pts]
                critical_pts.extend(cpts)
                for cp in cpts:
                    base_pts = self.critical_operator.find_base_points(critical_pts=[cp])
                    if len(base_pts) > 0:
                        base1 = base_pts[0][0]
                        base2 = base_pts[0][1]
                        cv.line(self.sweep_seg_map, (base1[1], base1[0]), (base2[1], base2[0]), OCCUPIED, thickness=2)

            # # debug
            # _, axs = plt.subplots(1, 1, constrained_layout=True)
            # ax = axs
            # ax.imshow(self.map_ws, cmap='gray', alpha=0.5)
            # ax.plot(xys[:, 1], xys[:, 0], 'r--', linewidth=4, label='Original Points')
            # ax.plot(xys_s_2[:, 1], xys_s_2[:, 0], 'c-', linewidth=3, label='Smoothed Points2')
            # for cp in cpts:
            #     ax.plot(cp.coord[1], cp.coord[0], 'ro', markersize=8, label='Critical Point')
            # ax.set_aspect('equal', adjustable='box')
            # ax.set_title(f'Branch {i} - Sweep Skeleton Decomposition')
            # ax.legend()
            # plt.show()

        return critical_pts

    def find_max_obst_dist_point(self, p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
        ''' Given points p1 and p2, find the distances of occupied points in the line on p1 and perpendicular to p1-p2.
        Return the point with maximum distance to obstacles. '''

        if p1[0] == p2[0] and p1[1] == p2[1]:
            return None

        th_p12 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])  # angle of line p1-p2
        max_l = max(self.sweep_seg_map.shape)         # maximum length to check
        dists = []
        for th in [th_p12 + np.pi / 2, th_p12 - np.pi / 2]:
            for d in range(0, max_l):
                x = int(p1[0] + d * np.cos(th))
                y = int(p1[1] + d * np.sin(th))
                if x < 0 or x >= self.sweep_seg_map.shape[0] or y < 0 or y >= self.sweep_seg_map.shape[1]:
                    raise ValueError(f"Point ({x}, {y}) is out of bounds for map shape {self.sweep_seg_map.shape}.")
                    # dists.append(d)
                    # break
                if self.sweep_seg_map[x, y] == OCCUPIED:
                    dists.append(d)
                    break
        if len(dists) < 2:
            # self.lg.error(f"Not enough occupied points found")
            # return None
            raise ValueError(f"Not enough occupied points.")
        return max(dists)

    def smooth_path(self, path: np.ndarray, smoothing_factor: float = 1.0, samples=None) -> np.ndarray:

        # remove duplicate points
        path = np.unique(path, axis=0)

        if samples is None:
            samples = len(path)

        x = path[:, 0]
        y = path[:, 1]
        tck, u = splprep([x, y])
        u_new = np.linspace(0, 1, samples)

        return np.column_stack(splev(u_new, tck))

    # ==================================================================================================================
    # Critical points based on local minimum distance to obstacles along branches
    # ==================================================================================================================

    def find_min_dist_critical_points(self) -> list[CriticalPoint]:

        critical_pts = []

        # grad_ep_thresh = 2 * self.robot_radius_px       # threshold for endpoint distance to consider gradient for critical point detection
        space_thresh = 4 * self.robot_radius_px         # threshold for minimum distance between critical points along the branch
        sigma_smooth = max(int(self.robot_radius_px * 2.0), 5)     # smoothing parameter for Gaussian filter

        for i, branch in enumerate(self.branches):

            n = len(branch.points)
            if n < (4 * self.robot_radius_px + 1):
                self.lg.error(f"Branch {i} is too short ({n} points). Skipping critical point detection.")
                continue

            dists = self.skelet.distance_map_norm[branch.points[:, 0], branch.points[:, 1]].astype(float)
            dist_in = dists
            sigma_ = sigma_smooth

            # filter
            f_d = gaussian_filter1d(dist_in, sigma=sigma_, radius=sigma_ * 2)
            f_d = gaussian_filter1d(f_d, sigma=sigma_)

            smooth = f_d
            smooth = self.filtar_branch_distances(smooth)
            # grad = np.gradient(smooth)

            # Find critical points based on corridors (flat regions) in the distance profile
            cpts = self.find_corridors(branch.points, smooth, space_thresh)
            if len(cpts) == 0:

                # # Check if the gradient is monotonic in the middle part of the branch (excluding endpoints)
                # grads_tr = grad[grad_ep_thresh:-grad_ep_thresh]
                # if all(grads_tr > 0) or all(grads_tr < 0):
                #     continue

                # Find critical points based on other methods (global or local minimum distance to obstacles)
                # cpts = self.find_min_dist_critical_points_global(branch.points, smooth)
                cpts = self.find_min_dist_critical_points_local(branch.points, smooth)

            critical_pts.extend(cpts)

        return critical_pts

    def filtar_branch_distances(self, dists: np.ndarray) -> np.ndarray:
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
        # d_max = np.max(dists)
        # dl = max(0.1 * (d_max - d_min), 1)
        # d_corr = d_min + dl
        if d_min > (0.2 * n):
            return []

        min_cor_len = int(max(8 * self.robot_radius_px, n * 0.45))

        # dists = dists / np.max(dists)  # normalize distances to [0, 1]
        # grads = np.gradient(dists)

        dist_diff_thresh = max(2, n / 200)

        cors = []
        start_d = dists[1]
        start = 1
        i = 1
        while i < n - 1:
            # if dists[i] < d_corr:
            # if abs(grads[i]) < 0.004:
            if abs(dists[i] - start_d) > dist_diff_thresh:
                end = i
                if end - start + 1 >= min_cor_len:
                    start_d = np.mean(dists[start:end])
                    cors.append((start, end))
                start_d = dists[i]
                start = i
            i += 1
        if i - 1 - start >= min_cor_len:
            cors.append((start, i - 1))

        def compute_angle(p1, p2, p3):
            """Compute the angle at p2 formed by the line segments p1-p2 and p2-p3."""
            v1 = np.array(p1) - np.array(p2)
            v2 = np.array(p3) - np.array(p2)
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # clip to avoid numerical issues
            return np.degrees(angle)

        # now for each start point of corridor, find the two base points by and checking if b1-cp-b2 angle is close to 180 degrees
        k = 5  # degree
        new_cors = []
        for start, end in cors:
            new_start = start
            new_end = end
            while new_start < end:
                temp_cp = CriticalPoint(coord=bpts[new_start], type='corridor')
                bases = self.critical_operator.find_base_points(critical_pts=[temp_cp])
                if len(bases) > 0:
                    bases = bases[0]
                    theta = compute_angle(bases[0], temp_cp.coord, bases[1])
                    if abs(theta - 180) < k:
                        break
                new_start += 1
            while new_end > start:
                temp_cp = CriticalPoint(coord=bpts[new_end], type='corridor')
                bases = self.critical_operator.find_base_points(critical_pts=[temp_cp])
                if len(bases) > 0:
                    bases = bases[0]
                    theta = compute_angle(bases[0], temp_cp.coord, bases[1])
                    if abs(theta - 180) < k:
                        break
                new_end -= 1
            if new_end - new_start + 1 >= min_cor_len:
                new_cors.append((new_start, new_end))

        # # nor for each corridor, find mean and std
        # # then update start and end to be the points where the distance is within mean +/- std
        # new_cors = []
        # for start, end in cors:
        #     mean_d = np.mean(dists[start:end + 1])
        #     std_d = np.std(dists[start:end + 1])
        #     d_thresh = max(2, std_d / 2)
        #     new_start = start
        #     new_end = end
        #     while new_start < end and abs(dists[new_start] - mean_d) > d_thresh:
        #         new_start += 1
        #     while new_end > start and abs(dists[new_end] - mean_d) > d_thresh:
        #         new_end -= 1
        #     if new_end - new_start + 1 >= min_cor_len:
        #         new_cors.append((new_start, new_end))

        cps = []
        for cor in new_cors:
            start, end = cor
            cps.append(CriticalPoint(coord=bpts[start], type='corridor'))
            cps.append(CriticalPoint(coord=bpts[end], type='corridor'))
        return cps

    def find_min_dist_critical_points_global(self, bpts: list[tuple], dists: np.ndarray) -> list[CriticalPoint]:
        """Find critical points based on local minimum distance to obstacles along branches."""
        cr_pts = []
        robot_radius_px = self.robot_radius_px
        search_r_px = 4 * robot_radius_px  # param # search radius for local minimum distance
        # search_r_px_2 = search_r_px // 2

        min_dist_idx = np.argmin(dists)
        min_dist = dists[min_dist_idx]
        max_dist = np.max(dists)

        mean_dist = np.mean(dists)

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

        n = len(bpts)
        search_r = max(8 * self.robot_radius_px, n // 4)  # param # search radius for local minimum distance
        i_skip = self.robot_radius_px

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

            if min_dist_local > 0.85 * mean_dist_local:  # param
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
        if len(self.critical_pts) > 0:
            cp_coords = np.array([cp.coord for cp in self.critical_pts])
            ax.scatter(cp_coords[:, 1], cp_coords[:, 0], color='orange', s=marker_size, label='Critical Points')
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
            ax2.text(xx[len(xx) // 2], yy[len(yy) // 2], f'B{i}', color='black', fontsize=8, ha='center', va='center', zorder=3)

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

        # plt.show()
        plt.close('all')

    @staticmethod
    def viz_img(img: np.ndarray, name: str):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
