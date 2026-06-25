"""Voronoi-based segmentation for optimal workspace decomposition."""
import heapq
from collections import defaultdict
import cv2 as cv
import numpy as np
from networkx import Graph
from matplotlib import pyplot as plt
from shapely.geometry import Polygon  # , LineString
from skimage.morphology import medial_axis  # , skeletonize
from covpp.base.logger import MyLogger
from covpp.base.plot_styles import set_plot_styles
from covpp.base.configs import RobotConfig, PipeLineConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.skeleton_segmentation.segmentation_improvement import SegmentationImprovement
from covpp.segmentation.skeleton_segmentation.data_structures import Skelet, Branch, CriticalPoint, SegmentSolution, is_valid_contour
# from tsmoothie.smoother import LowessSmoother

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

        # for debugging
        self.refinement_step = False

        # data
        self.zmd = zmd
        self.robot_radius_px = int(RobotConfig.radius / zmd.resolution)
        self.map_ws = zmd.map_ws   # uint8 binary map (OCCUPIED, FREE)

        # settings
        # threshold ratio for min distance from ends # [m1-3: 0.05][m4-m10:0.005][m11:0.00]
        self.end_dist_threshold_ratio: float = 0.0  # 0.05   # minimum ratio of distance from branch ends to overall min-max distance in branch to consider critical point
        self.max_critical_region_ratio: float = 0.6  # 0.6     # maximum ratio of critical region length to branch length
        self.min_max_distance_ratio: float = 0.6    # 0.8    # minimum ratio of min distance to max distance in branch to consider critical point
        self.min_area_thresh_for_contour: int = 10  # 10     # minimum area threshold for valid contour

        # attributes
        self.erp_sidep = []                              # for debugging
        self.error_point2 = []                              # for debugging
        self.skelet: Skelet = None                          # skeletonization data
        self.sk_graph: Graph = None                         # skeleton graph
        self.branches: list[Branch] = []                    # skeleton branches
        self.bpts_graph: Graph = None                       # graph of branch points
        self.cps: list[CriticalPoint] = []                  # critical points data
        self.base_points: list[tuple[tuple, tuple]] = []    # base points for critical points (n_cp * 2)

        # results
        self.polygons: list[Polygon] = []                   # shapely polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray      # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray      # map_ws with segmentation lines drawn

    def skeleton_decomposition(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        # create skeleton and extract special points and branches
        self.create_skeleton()
        self.find_endpoints_branchpoints()
        self.extract_special_branches()

        # find critical points and base points
        self.find_critical_points()
        self.find_base_points()
        # self.lg.info2(f"Found {len(self.cps)} critical points in skeleton branches.")
        # self.lg.info2(f"Found {len(self.base_points)} base points for critical points.")

        # create segmented map
        segment_solution = self.create_segmented_map()

        # second level segmentation (Improvement)
        segment_solution = self.improvement_step(segment_solution)

        # create adjacency matrix and polygons
        self.adjacency_matrix = self.create_adjacency_matrix(segment_solution)
        self.create_polygons(segment_solution.polygons)
        self.lg.info2(f"Extracted {len(self.polygons)} polygons.")

        self.visualize_skeleton()

    def improvement_step(self, segment_solution: SegmentSolution) -> SegmentSolution:
        segmentation_improver = SegmentationImprovement(segment_solution=segment_solution,
                                                        bpts_graph=self.bpts_graph,
                                                        map_ws_segmented=self.map_ws_segmented,
                                                        robot_radius_px=self.robot_radius_px)

        segmentation_improver.run()
        self.cps.extend(segmentation_improver.new_cps)
        for cp in segmentation_improver.new_cps:
            self.get_base_points_by_expand_side_points(cp)
        return self.create_segmented_map()

    # ==================================================================================================================
    # segmentation methods
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
        for bases in self.base_points:
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
            if not is_valid_contour(h, hierarchy):
                continue

            # minimum area threshold
            area = cv.contourArea(c)
            if area >= self.min_area_thresh_for_contour:
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

        # # visualize segmented map # debugging # todo
        # cv.imshow("Segmented Map", seg_map)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

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
        canvas_size = 40 // 2

        for cp in self.cps:
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
                self.error_point2.append(cp.coord)
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
        """Create skeleton of the workspace and find end_points and branch_points."""

        # Compute medial axis (skeleton) and the distance transform
        # ske: dtype=bool, distance: dtype=float64
        medial_axis_map, distance_map = medial_axis(self.map_ws, return_distance=True)
        skeleton_map = medial_axis_map

        # # visualize medial axis and distance map # debugging # todo
        # skeletonize_map = skeletonize(self.map_ws)  # dtype=bool
        # cv.namedWindow("skeletonize_map", cv.WINDOW_NORMAL)
        # cv.imshow("skeletonize_map", skeletonize_map.astype(np.uint8) * 255)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # distance transform and normalization
        # distance_map = cv.distanceTransform(self.map_ws, distanceType=cv.DIST_L2, maskSize=5)
        distance_map = distance_map.astype(np.uint8)
        distance_norm_map = (distance_map / distance_map.max() * FREE).astype(np.uint8)  # dtype=uint8

        # # skeletonization
        # skeleton_map = skeletonize(self.map_ws)  # dtype=bool
        # skeleton_lee = skeletonize(self.map_ws, method='lee')

        # distance_normalized masked by skeleton
        # distance_skeleton_map = np.zeros_like(distance_norm_map)  # dtype=uint8
        # distance_skeleton_map[skeleton_map] = distance_norm_map[skeleton_map]
        distance_skeleton_map = distance_norm_map * skeleton_map

        # ----------------------------------------------------------------------
        # Skeleton Segmented Map Creation
        skeleton_segmented_map = self.map_ws.copy()                 # start with workspace map
        skeleton_segmented_map[skeleton_map] = OCCUPIED             # set skeleton pixels to OCCUPIED
        skeleton_segmented_map = FREE - skeleton_segmented_map      # invert map, so skeleton contours are closed areas
        # find contours and fill areas
        contours2, _ = cv.findContours(skeleton_segmented_map, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        # fill area with different gray value
        if len(contours2) > 254:
            self.lg.warn(f"Too many contours found: {len(contours2)}.")
            skeleton_segmented_map = skeleton_segmented_map.astype(np.uint16)   # use uint16 to support more contours
        for i, cnt in enumerate(contours2):
            cv.fillPoly(skeleton_segmented_map, pts=[cnt], color=i + 1)
        skeleton_segmented_map[skeleton_map] = OCCUPIED                         # make skeleton pixels OCCUPIED again

        # # visualize skeleton segmented map # debugging # todo
        # cv.namedWindow("Skeleton Segmented Map", cv.WINDOW_NORMAL)
        # cv.imshow("Skeleton Segmented Map", skeleton_segmented_map.astype(np.uint8))
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # final skeletonization data
        self.skelet = Skelet(
            skeleton_map=skeleton_map,
            distance_map=distance_map,
            distance_norm_map=distance_norm_map,
            distance_skeleton_map=distance_skeleton_map,
            skeleton_segmented_map=skeleton_segmented_map,
            end_points=None,
            branch_points=None,
            end_points_filtered=None
        )

    def find_endpoints_branchpoints(self) -> None:
        """ Find end_points and branch_points in the skeleton. """

        skeleton = self.skelet.skeleton_map.astype(np.uint8)    # dtype=uint8

        # Convolution kernel to count neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)
        filtered = cv.filter2D(skeleton, -1, kernel)            # dtype=uint8

        # Identify end_points and branch_points based on neighbor counts
        end_points_map = np.where(filtered == 11, 1, 0).astype(np.uint8)     # dtype=uint8
        branch_points_map = np.where(filtered >= 13, 1, 0).astype(np.uint8)  # dtype=uint8

        # filtering end_points too close to branch_points
        mask = np.zeros_like(self.zmd.map_ws, dtype=np.uint8)
        branchpoint_coords = np.column_stack(np.where(branch_points_map))
        for y, x in branchpoint_coords:
            radius = 1.2 * self.skelet.distance_norm_map[y, x]
            cv.circle(mask, (x, y), int(radius), 1, thickness=-1)   # filled circle
        end_points_filtered_1 = end_points_map * (1 - mask)

        # second method for filtering
        # filter end_points with same distance as min_distance in skeleton
        distance_skeleton = self.skelet.distance_skeleton_map
        dists_sorted = np.sort(distance_skeleton[distance_skeleton > 0])
        dists_sorted = dists_sorted[:np.sum(end_points_map)]  # only need as many as end points
        dist_thresh = dists_sorted[-1]
        end_points_filtered2 = np.where(distance_skeleton > dist_thresh, end_points_map, 0).astype(np.uint8)

        # combine both filtering methods (AND)
        end_points_filtered_map = end_points_filtered_1 * end_points_filtered2

        # get coordinates
        end_points = np.column_stack(np.where(end_points_map))
        branch_points = np.column_stack(np.where(branch_points_map))
        end_points_filtered = np.column_stack(np.where(end_points_filtered_map))

        # validate
        if len(end_points) == 0:
            self.lg.error("No end points found in skeleton.")
        if len(branch_points) == 0:
            self.lg.error("No branch points found in skeleton.")

        # skeletonization data
        self.skelet.end_points = end_points
        self.skelet.branch_points = branch_points
        self.skelet.end_points_filtered = end_points_filtered

    def extract_special_branches(self) -> None:
        """ Extract branches from a skeleton as paths between end_points/branch_points. """
        self.create_graph_and_specials()
        self.branches = self.extract_special_branches_from_graph()

        # validate
        if len(self.branches) == 0:
            self.lg.warn("No branches extracted from skeleton.")

    # ==================================================================================================================
    # find critical points and base points
    # ==================================================================================================================

    def find_critical_points(self):
        """ Find local minimum distance points in each branch.
        If there are multiple points (next to each other) in the branch with the same minimum distance, select the middle one.
        """
        cp_pts: list[np.ndarray] = []  # list of (y, x) critical points
        cps: list[CriticalPoint] = []
        for branch in self.branches:
            dists = self.skelet.distance_norm_map[branch.points[:, 0], branch.points[:, 1]]
            n_dists = len(dists)

            # evaluate distances from branch ends and overall min/max
            # skip if min distance is close to end distances
            max_dist = dists.max()
            min_dist = dists.min()
            min_ind = np.argmin(dists)
            dist_del = max_dist - min_dist

            # skip if min distance is at the ends of the branch
            if min_ind == 0 or min_ind == n_dists - 1:
                continue
            # skip if min distance is close to end distances
            if self.end_dist_threshold_ratio > 0:
                min_end_dist = min(dists[0], dists[-1])
                d_min_end = min_end_dist - min_dist
                if d_min_end < self.end_dist_threshold_ratio * dist_del:
                    continue

            # evaluate length of min_indices
            min_dist_thresh = min_dist + 0.1 * dist_del
            min_indices = np.where(dists < min_dist_thresh)[0]
            if len(min_indices) / branch.length > self.max_critical_region_ratio:
                continue

            # find consecutive indices with distance less than a threshold (to handle flat regions with same min distance)
            min_idx = np.argmin(dists)
            upper_idx = min_idx + 1
            lower_idx = min_idx - 1
            while (upper_idx < n_dists and dists[upper_idx] < (min_dist_thresh)):
                upper_idx += 1
            while (lower_idx >= 0 and dists[lower_idx] < (min_dist_thresh)):
                lower_idx -= 1
            min_indices = np.arange(lower_idx + 1, upper_idx)

            # only consider if min_dist is significantly less than max_dist
            if min_dist > max_dist * self.min_max_distance_ratio:
                continue

            # select middle index if multiple
            mid_idx = min_indices[len(min_indices) // 2]
            # if len(min_indices) > 1:
            #     mid_idx = min_indices[len(min_indices) // 2]
            # else:
            #     mid_idx = min_indices[0]

            # take the first one and last one (to avoid biasing to one end) # todo
            # min_indices = np.array([min_indices[0], min_indices[-1]])
            # min_indices = np.array([min_indices[0]])
            min_indices = np.array([mid_idx])
            local_ps = branch.points[min_indices]
            cp_pts.extend(local_ps)
            cps.extend([CriticalPoint(coord=lp) for lp in local_ps])

        # update critical points list
        self.cps.extend(cps)

    def find_base_points(self):
        """Find base points for all critical points."""

        for cp in self.cps:
            self.get_base_points_by_expand_side_points(cp)

        # validate
        if len(self.base_points) != len(self.cps):
            self.lg.warn(f"N base points {len(self.base_points)} != N critical points {len(self.cps)}.")
            # raise ValueError("Base points and critical points count mismatch.")

    # ==================================================================================================================
    # find base points methods
    # ==================================================================================================================

    def get_base_points_by_expand_side_points(self, cp: CriticalPoint, n_pts: int = 2) -> bool:
        """ Find two side points for a critical point and expand until finding base points.

        Expansion is guided by `distance_map`: neighbors with lower distance are explored first,
        so the search tends toward walls/contours faster.
        """

        dxy_8c = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # find side points
        if not self.find_side_points(cp, n_pts=n_pts):
            return False

        # expand like graph search side points until hitting contour or occupied space (stop expansion if reached skeleton)
        visited: set[tuple] = set()
        base_pts: list[tuple[int, int]] = []            # base points
        base_paths: list[tuple[tuple[int, int]]] = []   # base paths

        # for each side point, expand until hitting contour or occupied space
        for side_pt, sk_segment in zip(cp.side_points, cp.side_sk_segments):
            base_path = []
            heap: list[tuple[int, int, int, int]] = []  # (priority=distance, steps, y, x)
            sy, sx = side_pt[0], side_pt[1]
            heapq.heappush(heap, (self.skelet.distance_norm_map[sy, sx], 0, sy, sx))

            base_p = None
            while heap and base_p is None:
                _, steps, cy, cx = heapq.heappop(heap)
                if (cy, cx) in visited:
                    continue
                visited.add((cy, cx))
                base_path.append((cy, cx))

                # check if hitting contour or occupied space
                if self.map_ws[cy, cx] == 0:
                    base_p = (cy, cx)
                    break

                # if on skeleton, do not expand further
                if self.skelet.skeleton_map[cy, cx]:
                    continue

                # check if in the same skeleton segment
                sk_seg = self.skelet.skeleton_segmented_map[cy, cx]
                if sk_seg != sk_segment:
                    continue

                # expand to neighbors prioritized by lower distance
                for ddy, ddx in dxy_8c:
                    ny, nx = cy + ddy, cx + ddx
                    if (ny, nx) in visited:
                        continue
                    nprio = self.skelet.distance_map[ny, nx]
                    new_steps = steps + 1
                    if ddy != 0 and ddx != 0:
                        new_steps += 1  # diagonal penalty
                    heapq.heappush(heap, (nprio, new_steps, ny, nx))

            if base_p is not None:
                base_pts.append(base_p)
                base_paths.append(tuple(base_path))

        # validate results
        if len(base_pts) == 0:
            # raise ValueError(f"No base points found for critical point at {cp}.")
            self.lg.error(f"No base points found for critical point at {cp}.")
            return False
        if len(base_pts) != n_pts:
            # raise ValueError(f"Expected {n_pts} base points, found {len(base_pts)} for critical point at {cp}.")
            self.lg.error(f"Expected 2 base points, found {len(base_pts)} for critical point at {cp}.")
            return False

        # store results
        cp.base_points = base_pts
        cp.base_paths = base_paths
        self.base_points.append(tuple(base_pts))
        return True

    def find_side_points(self, cp: CriticalPoint, n_pts: int = 2) -> bool:
        """ Find side points for a critical point. 

        Side points are found by expanding from the critical point to neighboring pixels.
        Side points must be in different skeleton segments than each other and the critical point.

        Steps:
        - Start from the critical point and explore its 8-connectivity neighbors.
        - For each neighbor, check its skeleton segment.
        - If the neighbor's skeleton segment is different from the critical point's segment and not already found,
          add it to the side points list.
        - Continue exploring until the required number of side points is found.
        """

        # find neighbors (8-connectivity) of cp
        dxy_8c = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        cp_coord = cp.coord
        cp_sk_segment = self.skelet.skeleton_segmented_map[cp_coord[0], cp_coord[1]]
        sk_segments = [OCCUPIED, cp_sk_segment]  # to ignore skeleton
        open = [cp_coord]
        visited: set[tuple] = set()
        side_pts: list[tuple[int, int]] = []

        while open and len(side_pts) < n_pts:
            p = open.pop(0)
            cy, cx = p
            if (cy, cx) in visited:
                continue
            visited.add((cy, cx))
            for dy, dx in dxy_8c:
                y = cy + dy
                x = cx + dx
                sk_seg = self.skelet.skeleton_segmented_map[y, x]

                # had to add this check to find side points in cases where cp more surrounded by skeleton
                # add skeleton segment to open list to explore further around it
                if sk_seg == OCCUPIED:
                    open.append((y, x))
                    continue

                if sk_seg in sk_segments:
                    continue

                sk_segments.append(sk_seg)
                side_pts.append((y, x))

            # only want to explore a few layers away from cp
            if len(open) > 4:
                break

        # # side points
        # # plot for debugging # todo
        # if self.refinement_step:
        #     _, ax = plt.subplots(1, 1)
        #     ax.imshow(self.skelet.skeleton_segmented_map, cmap='nipy_spectral')
        #     ax.scatter(cp_coord[1], cp_coord[0], c='r', marker='x', s=50)
        #     for sp in side_pts:
        #         ax.scatter(sp[1], sp[0], c='b', marker='o', s=25)
        #     ax.set_aspect('equal')
        #     plt.show()

        # validate
        if len(side_pts) != n_pts:
            self.erp_sidep.append(cp_coord)
            self.lg.warn(f"Expected {n_pts} side points, found {len(side_pts)} for critical point at {cp_coord}.")
            return False
            # raise ValueError(f"Expected {n_pts} side points, found {len(side_pts)} for critical point at {cp_coord}.")

        # store results in cp
        cp.cp_skel_segment = cp_sk_segment
        cp.side_sk_segments = sk_segments[2:]
        cp.side_points = side_pts
        return True

    # ==================================================================================================================
    # Branch extraction method
    # ==================================================================================================================

    def create_graph_and_specials(self) -> Graph:
        """ Create graph from skeleton and identify special nodes (end_points and branch_points). """

        # Build graph
        graph = Graph()
        height, width = self.skelet.skeleton_map.shape
        coords: np.ndarray = np.column_stack(np.where(self.skelet.skeleton_map))
        for y, x in coords:
            graph.add_node((y, x))

        # Neighbor offsets
        # nbrs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Add edges
        skel_set = set(map(tuple, coords))
        for y, x in coords:
            for dy, dx in nbrs:
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx_ < width and (ny, nx_) in skel_set:
                    # Undirected edge; NetworkX avoids duplicates
                    graph.add_edge((y, x), (ny, nx_))

        # validate
        if graph.number_of_nodes() < 20:  # parameter
            raise ValueError("Skeleton graph has too few nodes, likely invalid skeleton.")
        self.sk_graph = graph

    def extract_special_branches_from_graph(self) -> list[Branch]:
        """ Extract branches from the graph between special nodes or end_points. """

        # Traverse branches:
        # from each special node along degree-2 nodes to next special or endpoint
        visited_edges = set()
        branches = []

        # graph to store branch points connections
        bps_graph = Graph()

        # Identify special nodes (end_points_filtered and branch_points)
        branch_pts = set(map(tuple, self.skelet.branch_points))
        ent_pts = set(map(tuple, self.skelet.end_points_filtered))
        special_pts = ent_pts.union(branch_pts)

        #
        n_branches = 0
        be_pts_nb = defaultdict(int)  # n branches connected to each endpoints of branches
        be_pts_idx_b = defaultdict(list)  # index of branch for each endpoint of branches

        # Start walks from each special node
        for s in special_pts:
            for v in self.sk_graph.neighbors(s):
                e = tuple(sorted((s, v)))
                if e in visited_edges:
                    continue
                # Walk until next special/junction
                path: list[tuple] = self.walk_branch(s, v, special_pts, self.sk_graph)
                # Mark all edges in the path as visited
                for a, b in zip(path[:-1], path[1:]):
                    visited_edges.add(tuple(sorted((a, b))))
                start = path[0]
                end = path[-1]
                points = np.array(path, dtype=int)

                # Classify end type
                if end in special_pts:
                    end_type = 'special'
                else:
                    end_type = 'skip'
                    continue

                # update branch points graph
                if start in branch_pts and end in branch_pts:
                    bps_graph.add_edge(start, end)
                    dist = np.linalg.norm(np.array(start) - np.array(end))
                    bps_graph[start][end]['distance'] = dist
                    # bps_graph[start][end]['type'] = ''

                # check branch length
                len_branch = len(points)
                if len_branch < 4 * self.robot_radius_px:
                    continue
                    # end_type = 'skip'

                # update branch points count for endpoints
                be_pts_nb[start] += 1
                be_pts_nb[end] += 1
                be_pts_idx_b[start].append(n_branches)
                be_pts_idx_b[end].append(n_branches)

                # collect branch info
                branches.append(Branch(points=points,
                                       start_point=start,
                                       end_point=end,
                                       end_type=end_type,
                                       length=len_branch,
                                       ))
                n_branches += 1

        self.bpts_graph = bps_graph
        return branches

        # new_branches = self.filter_branches(branches, be_pts_nb, be_pts_idx_b)
        # return new_branches

    def walk_branch(self, start: tuple, nxt: tuple, specials: set[tuple], graph: Graph) -> list[tuple]:
        """ Walk along a branch from start to next special node or endpoint."""
        path = [start, nxt]
        prev = start
        curr = nxt
        while True:
            # deg = graph.degree[curr]
            if curr in specials and curr != start:
                break
            # Continue through degree-2 chain
            nbrs = list(graph.neighbors(curr))
            # Exclude where we came from
            forward = [u for u in nbrs if u != prev]
            if len(forward) != 1:
                # Dead-end or junction without being marked special
                break
            nxt2 = forward[0]
            path.append(nxt2)
            prev, curr = curr, nxt2
        return path

    # def filter_branches(self, branches: list[Branch], be_pts_nb: dict[tuple, int], be_pts_idx_b: dict[tuple, list[int]]) -> list[Branch]:
    #     print(f"Filtering branches. Initial count: {len(branches)}")
    #     # find branche pairs shared endpoints with nb = 2 and merge them into one
    #     merged_idxs = []
    #     new_branches = []
    #     for pt, nb in be_pts_nb.items():
    #         if nb == 2:
    #             idx_b1, idx_b2 = be_pts_idx_b[pt]
    #             if idx_b1 in merged_idxs or idx_b2 in merged_idxs:
    #                 continue
    #             b1 = branches[idx_b1]
    #             b2 = branches[idx_b2]
    #             # merge b1 and b2 into one branch
    #             if b1.start_point == pt:
    #                 points1 = b1.points[::-1]
    #             else:
    #                 points1 = b1.points
    #             if b2.start_point == pt:
    #                 points2 = b2.points[1:]
    #             else:
    #                 points2 = b2.points[:-1][::-1]
    #             merged_points = np.concatenate([points1, points2], axis=0)
    #             merged_branch = Branch(points=merged_points,
    #                                    start_point=merged_points[0],
    #                                    end_point=merged_points[-1],
    #                                    end_type='special',
    #                                    length=len(merged_points),
    #                                    )
    #             new_branches.append(merged_branch)
    #             merged_idxs.extend([idx_b1, idx_b2])
    #             print(f"Merged branches {idx_b1} and {idx_b2} at point {pt}.")
    #     # add branches that are not merged
    #     for i, b in enumerate(branches):
    #         if i not in merged_idxs:
    #             new_branches.append(b)
    #     return new_branches

    # ==================================================================================================================
    # visualization methods
    # ==================================================================================================================

    def visualize_skeleton(self) -> None:
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

        # data
        branches = self.branches
        end_pts = self.skelet.end_points
        branch_pts = self.skelet.branch_points
        end_pts_filtered = self.skelet.end_points_filtered

        # professional styles
        fig_w = 8.27  # A4 width in inches
        fig_h = fig_w * self.zmd.hw[0] / self.zmd.hw[1]
        set_plot_styles(fig_w, fig_h)

        # scale_m: float = min(zmd.hw)*zmd.resolution/10
        # A4 size: 8.27 x 11.69 inches
        scale_px = min(self.zmd.hw) / 500.0
        marker_size = 50 / scale_px
        colors = plt.cm.get_cmap('hsv', len(branches))

        # --------------------------------------------------------------------------------------------------------------
        # Figure 1: skeleton and endpoints/branchpoints
        fig1, ax = plt.subplots(1, 1, constrained_layout=True)

        # plot skeleton (medial) axis - endpoints and branchpoints in same plot
        ax.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=2 / scale_px)
        ax.imshow(FREE - self.skelet.distance_skeleton_map, cmap='bone', alpha=1.0)
        # endpoints
        ax.scatter(end_pts[:, 1], end_pts[:, 0], color='blue', s=marker_size, label='Endpoints')
        # branchpoints
        ax.scatter(branch_pts[:, 1], branch_pts[:, 0], color='red', s=marker_size, label='Branchpoints')
        # filtered endpoints
        ax.scatter(end_pts_filtered[:, 1], end_pts_filtered[:, 0], color='lime', s=marker_size, label='Filtered Endpoints')
        configure_ax(ax, 'Skeleton with Endpoints and Branchpoints', show_title=False)

        # --------------------------------------------------------------------------------------------------------------
        # Figure 2: plot branches with critical points and base points
        fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)

        # Draw contours in the background
        ax2.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=4 / scale_px, zorder=0)
        # branches
        for i, branch in enumerate(branches):
            label = "Branch" if i == 0 else None
            yy, xx = branch.points[:, 0], branch.points[:, 1]
            color = 'orange' if branch.end_type == 'skip' else colors(i)
            # Draw branch lines below points
            ax2.plot(xx, yy, color=color, linewidth=4 / scale_px, linestyle='-', label=label, zorder=2)

        # for all critical points
        for cp in self.cps:

            # critical point
            y, x = cp.coord
            ax2.scatter(x, y, color='red', alpha=1, s=marker_size, zorder=4, label='Critical Point' if cp == self.cps[0] else None)

            # base points
            for bp in cp.base_points:
                ax2.scatter(bp[1], bp[0], color='blue', alpha=1, s=marker_size, zorder=4, label='Base Point' if cp == self.cps[0] else None)

            # path to base points
            for path in cp.base_paths:
                path = np.array(path)
                ax2.plot(path[:, 1], path[:, 0], color='gray', linewidth=4 / scale_px, linestyle='-', zorder=1)

        # plot error point if any
        if self.erp_sidep:  # no side points found,  len(side_pts) != 2:
            for iep, ep in enumerate(self.erp_sidep):
                ax2.scatter(ep[1], ep[0], color='red', s=marker_size * 3, zorder=5, label='Error erp_sidep' if iep == 0 else None)
        if self.error_point2:  # less than 2 segments found, not enough to create adjacency
            for iep, ep in enumerate(self.error_point2):
                ax2.scatter(ep[1], ep[0], color='cyan', s=marker_size * 3, zorder=5, label='Error point2' if iep == 0 else None)

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
        # viz_map[self.skelet.skeleton_map] = 0  # hide skeleton for better visualization
        # # ti uint8 for visualization
        # if viz_map.dtype != np.uint8:
        #     viz_map = viz_map.astype(np.uint8)
        # skel_div_map_cp = cv.cvtColor(viz_map, cv.COLOR_GRAY2BGR)
        # for cp in self.cps:
        #     y, x = cp.coord
        #     skel_div_map_cp[y, x] = (0, 0, 255)
        # # show error points
        # for ep in self.erp_sidep:
        #     y, x = ep
        #     skel_div_map_cp[y, x] = (255, 0, 0)  # error points in blue
        # for ep in self.error_point2:
        #     y, x = ep
        #     skel_div_map_cp[y, x] = (255, 255, 0)  # error points in cyan
        # cv.imshow("Division Map with Critical Points", skel_div_map_cp)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
