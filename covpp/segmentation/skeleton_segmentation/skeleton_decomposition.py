"""Voronoi-based segmentation for optimal workspace decomposition."""
import heapq
import cv2 as cv
import numpy as np
from networkx import Graph
from matplotlib import pyplot as plt
from shapely.geometry import Polygon
from skimage.morphology import medial_axis  # , skeletonize
from covpp.base.logger import MyLogger
from covpp.base.plot_styles import set_plot_styles
from covpp.base.configs import RobotConfig, PipeLineConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.skeleton_segmentation.data_structures import Skelet, Branch
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

        # data
        # self.map_polygons = map_polygons
        self.zmd = zmd
        self.robot_radius_px = int(RobotConfig.radius / zmd.resolution)
        self.map_ws = zmd.map_ws   # uint8 binary map (OCCUPIED, FREE)

        # settings
        self.min_branch_length_px: int = 10             # minimum branch length to be considered
        self.end_dist_threshold_ratio: float = 0.1      # threshold ratio for min distance from ends
        self.max_critical_region_ratio: float = 0.6     # maximum ratio of critical region length to branch length
        self.min_max_distance_ratio: float = 0.8        # minimum ratio of min distance to max distance in branch to consider critical point

        # attributes
        self.error_point1 = []             # for debugging
        self.error_point2 = []             # for debugging
        self.skelet: Skelet = None          # skeletonization data
        self.branches: list[Branch] = []    # skeleton branches
        self.critical_points: list[np.ndarray] = []                 # local min distance points in branches (n_cp)
        self.base_points: list[tuple[tuple, tuple]] = []            # base points for critical points (n_cp * 2)
        self.side_points: list[tuple[tuple, tuple]] = []            # side points for critical points (n_cp * 2)
        self.cp_skel_segments: list[tuple[tuple, tuple]] = []       # division areas for each critical point (n_cp)
        self.base_paths: list[tuple[tuple, tuple]] = []             # connecting each base point to the critical point

        # results
        self.polygons: list[Polygon] = []                           # shapely polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray              # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray              # map_ws with segmentation lines drawn

    def skeleton_decomposition(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        self.create_skeleton()
        self.find_endpoints_branchpoints()
        self.extract_special_branches()

        self.find_critical_points()
        self.find_base_points()
        # self.lg.info2(f"Found {len(self.critical_points)} critical points in skeleton branches.")
        # self.lg.info2(f"Found {len(self.base_points)} base points for critical points.")

        segmented_map, seg_ids, contours = self.create_segmented_map()
        self.adjacency_matrix = self.create_adjacency_matrix(segmented_map, seg_ids)
        self.create_polygons(contours, self.adjacency_matrix)
        self.lg.info2(f"Extracted {len(self.polygons)} polygons.")

        self.visualize_skeleton()

    # ==================================================================================================================
    # segmentation methods
    # ==================================================================================================================

    def create_segmented_map(self):
        """ Create segmented map based on base points. 

        Steps:
        - Connect each two base points with a line.
        - Find contours and fill areas to create segmented map.
        - Each area will have a unique gray value. (later used in creating cells)
        """

        seg_map = self.map_ws.copy()

        # create a line connecting each two base points
        for bases in self.base_points:
            base1, base2 = bases
            cv.line(seg_map, (base1[1], base1[0]), (base2[1], base2[0]), OCCUPIED, thickness=2)

        # # # create map_ws_segmented
        self.map_ws_segmented = seg_map.copy()

        # put the base_paths on the map_ws and create segmented map
        # for paired_paths in self.base_paths:
        #     for path in paired_paths:
        #         for y, x in path:
        #             seg_map[y, x] = OCCUPIED

        # # canny edge detection for better contours
        # seg_map_canny = cv.Canny(seg_map, 100, 200)
        # seg_map = seg_map_canny

        seg_ids = []

        # find contours and fill areas
        contours, hierarchy = cv.findContours(seg_map, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        self.lg.debug(f"Found {len(contours)} contours in segmented map.")

        def find_level(h):
            """ Find the level of contour in hierarchy. """
            level = 0
            parent = h[3]
            while parent != -1:
                level += 1
                parent = hierarchy[parent][3]
            return level

        def is_valid_contour(h) -> bool:
            # if level is odd, it is a hole contour
            level = find_level(h)
            if level % 2 == 1:
                return False
            return True
            # # Check if contour has both parent and children in hierarchy.
            # return not(h[3] != -1 and h[2] != -1)

        # filter contours: remove those with children and small areas
        filtered_contours = []
        it = 0
        for c, h in zip(contours, hierarchy):
            if not is_valid_contour(h):
                continue  # skip contours with children
            area = cv.contourArea(c)
            if area >= 10:  # minimum area threshold # param
                filtered_contours.append(c)
                it += 1
        self.lg.debug(f"Found {len(filtered_contours)} filtered contours in segmented map.")

        # check too many contours
        if len(filtered_contours) > 254:
            self.lg.warn(f"Too many contours found in segmented map: {len(filtered_contours)}.")

        # fill area with different gray value (used for segment IDs)
        for i,  cnt in enumerate(filtered_contours):
            new_id = i + 1
            seg_ids.append(new_id)
            cv.fillPoly(seg_map, pts=[cnt], color=new_id)
            cv.drawContours(seg_map, [cnt], 0, FREE, thickness=1)

        # # visualize segmented map
        # cv.imshow("Segmented Map", seg_map)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        if not seg_ids:
            raise ValueError("No segments found in segmented map.")

        return seg_map, seg_ids, filtered_contours

    def create_adjacency_matrix(self, segmented_map: np.ndarray, seg_ids: list[int]) -> np.ndarray:

        adjacency_matrix = np.zeros((len(seg_ids), len(seg_ids)), dtype=np.bool)
        h, w = segmented_map.shape
        canvas_size = 40 // 2

        for cp in self.critical_points:
            y, x = cp
            y_min = max(0, y - canvas_size)
            y_max = min(h, y + canvas_size)
            x_min = max(0, x - canvas_size)
            x_max = min(w, x + canvas_size)
            canvas = segmented_map[y_min:y_max, x_min:x_max]
            unique_ids = np.unique(canvas)
            # remove OCCUPIED and FREE
            unique_ids = unique_ids[(unique_ids != OCCUPIED) & (unique_ids != FREE)]
            n_ids = len(unique_ids)
            if n_ids < 2:
                self.error_point2.append(cp)
                self.lg.warn(f"Less than 2 segments found near critical point at {cp}.")

            for i in range(n_ids):
                for j in range(i+1, n_ids):
                    adjacency_matrix[unique_ids[i]-1, unique_ids[j]-1] = True
                    adjacency_matrix[unique_ids[j]-1, unique_ids[i]-1] = True
        return adjacency_matrix

    def create_polygons(self, contours: list[np.ndarray], adjacency_matrix: np.ndarray) -> None:

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
            cv.fillPoly(skeleton_segmented_map, pts=[cnt], color=i+1)
        skeleton_segmented_map[skeleton_map] = OCCUPIED                      # make skeleton pixels OCCUPIED again

        # # visualize skeleton segmented map
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

        #
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
        graph = self.create_graph_and_specials()
        self.branches = self.extract_special_branches_from_graph(graph)

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
        for branch in self.branches:
            dists = self.skelet.distance_norm_map[branch.points[:, 0], branch.points[:, 1]]
            # smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)
            # smoother.smooth(dists.tolist())
            # dists_smooth = np.array(smoother.smooth_data[0])

            # evaluate distances from branch ends and overall min/max
            # skip if min distance is close to end distances
            min_end_dist = min(dists[0], dists[-1])
            max_dist = dists.max()
            min_dist = dists.min()
            d_min_max = max_dist - min_dist
            d_min_end = min_end_dist - min_dist
            if d_min_end < self.end_dist_threshold_ratio * d_min_max:
                continue

            min_indices = np.where(dists == min_dist)[0]

            # evaluate length of min_indices
            if len(min_indices)/branch.length > self.max_critical_region_ratio:
                continue

            # only consider if min_dist is significantly less than max_dist
            if min_dist > max_dist * self.min_max_distance_ratio:
                continue

            # # select middle index if multiple
            # if len(min_indices) > 1:
            #     mid_idx = min_indices[len(min_indices) // 2]
            # else:
            #     mid_idx = min_indices[0]

            # take the first one and last one (to avoid biasing to one end) # todo
            min_indices = np.array([min_indices[0], min_indices[-1]])
            # min_indices = np.array([min_indices[0]])
            local_ps = branch.points[min_indices]
            cp_pts.extend(local_ps)

            # update branch  info
            branch.cp_coords = local_ps
            branch.cp_idxs = min_indices

        # update critical points list
        self.critical_points = cp_pts

    def find_base_points(self):
        """Find base points for critical points."""

        for branch in self.branches:
            if branch.cp_coords is None:
                continue
            for cp in branch.cp_coords:
                self.get_base_points_by_expand_side_points(cp)

        # validate
        if len(self.base_points) != len(self.critical_points):
            self.lg.warn(f"N base points {len(self.base_points)} != N critical points {len(self.critical_points)}.")
            # raise ValueError("Base points and critical points count mismatch.")

        # ==================================================================================================================
        # find base points methods
        # ==================================================================================================================

    def get_base_points_by_expand_side_points(self, cp: np.ndarray) -> bool:
        """ Find two side points for a critical point and expand until finding base points.

        Expansion is guided by `distance_map`: neighbors with lower distance are explored first,
        so the search tends toward walls/contours faster.
        """

        # dxy_4c = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dxy_8c = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)]

        # find side points
        side_pts = self.get_side_points(cp)
        if side_pts is None:
            return False

        # expand like graph search two side points until hitting contour or occupied space (stop expansion if reached skeleton)
        base_pts: list[tuple[int, int]] = []            # two base points
        base_paths: list[tuple[tuple[int, int]]] = []   # two base paths
        for side_pt in side_pts:
            base_path = []
            visited: set[tuple] = set()
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

                # expand to neighbors prioritized by lower distance
                for ddy, ddx in dxy_8c:
                    ny, nx = cy + ddy, cx + ddx
                    if (ny, nx) in visited:
                        continue
                    nprio = int(self.skelet.distance_map[ny, nx])
                    new_steps = steps + 1
                    if ddy != 0 and ddx != 0:
                        new_steps += 1  # diagonal penalty
                    heapq.heappush(heap, (nprio, new_steps, ny, nx))

            if base_p is not None:
                base_pts.append(base_p)
                base_paths.append(tuple(base_path))

        # validate results
        if len(base_pts) == 0:
            # self.lg.error(f"No base points found for critical point at {cp}.")
            raise ValueError(f"No base points found for critical point at {cp}.")
        if len(base_pts) != 2:
            # self.lg.error(f"Expected 2 base points, found {len(base_pts)} for critical point at {cp}.")
            raise ValueError(f"Expected 2 base points, found {len(base_pts)} for critical point at {cp}.")

        # store results
        self.base_points.append(tuple(base_pts))
        self.base_paths.append(tuple(base_paths))
        return True

    def get_side_points(self, cp: np.ndarray) -> np.ndarray:
        """ Find two side points for a critical point. """

        # find neighbors (8-connectivity) of cp
        dxy_8c = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)]
        side_pts: list[tuple[int, int]] = []
        divide_values = [OCCUPIED]  # to ignore skeleton
        for dy, dx in dxy_8c:
            y, x = cp
            y += dy
            x += dx
            dv = self.skelet.skeleton_segmented_map[y, x]
            # ingore if on skeleton
            # if self.skelet.skeleton_map[y, x]:
            if dv in divide_values:
                continue
            divide_values.append(dv)
            side_pts.append((y, x))

        if len(side_pts) != 2:
            self.error_point1.append(cp)
            self.lg.warn(f"Expected 2 side points, found {len(side_pts)} for critical point at {cp}.")
            return None
            # raise ValueError(f"Expected 2 side points, found {len(side_pts)} for critical point at {cp}.")

        self.cp_skel_segments.append(tuple(divide_values[1:]))
        self.side_points.append(tuple(side_pts))
        return side_pts

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

        # Special nodes (degree 1 and >=3)
        # specials: set[tuple] = set()
        # # Fallback: classify by degree if masks are imperfect
        # for n in list(graph.nodes):
        #     deg = graph.degree[n]
        #     if deg == 1 or deg >= 3:
        #         specials.add(n)

        # validate
        if graph.number_of_nodes() < 20:  # parameter
            raise ValueError("Skeleton graph has too few nodes, likely invalid skeleton.")

        return graph

    def extract_special_branches_from_graph(self, graph: Graph) -> list[Branch]:
        """ Extract branches from the graph between special nodes or end_points. """

        # Traverse branches:
        # from each special node along degree-2 nodes to next special or endpoint
        visited_edges = set()
        branches = []

        # Identify special nodes (end_points_filtered and branch_points)
        special_pts = set(map(tuple, self.skelet.end_points_filtered)).union(set(map(tuple, self.skelet.branch_points)))

        # Start walks from each special node
        for s in special_pts:
            for v in graph.neighbors(s):
                e = tuple(sorted((s, v)))
                if e in visited_edges:
                    continue
                # Walk until next special/junction
                path: list[tuple] = self.walk_branch(s, v, special_pts, graph)
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

                # check branch length
                len_branch = len(points)
                if len_branch < max(self.min_branch_length_px, 4 * self.robot_radius_px):
                    end_type = 'skip'

                # collect branch info
                if end_type != 'skip':
                    branches.append(Branch(points=points,
                                           start_point=start,
                                           end_point=end,
                                           end_type=end_type,
                                           length=len_branch,
                                           ))
        return branches

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
                    fontsize='small',
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1),
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
        scale_px = min(self.zmd.hw)/500.0
        marker_size = 10 / scale_px
        colors = plt.cm.get_cmap('hsv', len(branches))

        # --------------------------------------------------------------------------------------------------------------
        # Figure 1: skeleton and endpoints/branchpoints
        fig1, ax = plt.subplots(1, 1, constrained_layout=True)

        # plot skeleton (medial) axis - endpoints and branchpoints in same plot
        ax.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=2/scale_px)
        ax.imshow(FREE-self.skelet.distance_skeleton_map, cmap='bone', alpha=1.0)
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

        ax2.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=2/scale_px)
        # branches
        for i, branch in enumerate(branches):
            label = "Branch" if i == 0 else None
            yy, xx = branch.points[:, 0], branch.points[:, 1]
            color = 'orange' if branch.end_type == 'skip' else colors(i)
            plt.plot(xx, yy, color=color, linewidth=2/scale_px, linestyle='-', label=label)

        # critical points
        critical_pts = np.array(self.critical_points).reshape(-1, 2)
        ax2.scatter(critical_pts[:, 1], critical_pts[:, 0], color='magenta', s=marker_size, label='Critical Points')

        # base points
        base_pts = np.array(self.base_points).reshape(-1, 2)
        ax2.scatter(base_pts[:, 1], base_pts[:, 0], color='green', s=marker_size, label='Base Points')

        # base paths
        labeled = True
        for paths in self.base_paths:
            for path in paths:
                path = np.array(path)
                label = "Bases Connection" if labeled else None
                ax2.plot(path[:, 1], path[:, 0], color='gray', linewidth=2/scale_px, linestyle='--', label=label)
                labeled = False

        # plot error point if any
        if self.error_point1:
            for ep in self.error_point1:
                ax2.scatter(ep[1], ep[0], color='red', s=marker_size*3, label='Error Point')
        if self.error_point2:
            for ep in self.error_point2:
                ax2.scatter(ep[1], ep[0], color='cyan', s=marker_size*3, label='Error Point 2')

        # finalize
        configure_ax(ax2, 'Detected Branches with Critical and Base Points', show_title=False, show_legend=False)
        ax2.invert_yaxis()

        # # --------------------------------------------------------------------------------------------------------------
        # # Figure 3: division map
        # _, ax3 = plt.subplots(1, 1, constrained_layout=True)
        # ax3.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=2/scale_px)
        # ax3.imshow(self.skelet.skeleton_segmented_map, cmap='tab20', alpha=1.0)
        # configure_ax(ax3, 'Division Map based on Skeleton', show_legend=False)

        # save figures
        im_format = 'svg'  # 'svg'
        if PipeLineConfig.do_save_decomposition_plot:
            save_dir = self.zmd.segmentation_method_dir
            fig1_path = save_dir + "/skeleton_end_branch_points" + f".{im_format}"
            fig2_path = save_dir + "/skeleton_branches_criticals" + f".{im_format}"
            fig1.savefig(fig1_path, format=im_format, dpi=PipeLineConfig.dpi, bbox_inches='tight')
            fig2.savefig(fig2_path, format=im_format, dpi=PipeLineConfig.dpi, bbox_inches='tight')

        # # plt.show()

        # # # visualize using OpenCV
        # cv.imshow("distance", self.skelet.distance_norm_map)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # # cv.imshow("Division Map", self.skelet.skeleton_segmented_map)
        # # show critical points
        # viz_map = self.skelet.skeleton_segmented_map.copy()
        # viz_map[self.skelet.skeleton_map] = 0  # hide skeleton for better visualization
        # # ti uint8 for visualization
        # if viz_map.dtype != np.uint8:
        #     viz_map = viz_map.astype(np.uint8)
        # skel_div_map_cp = cv.cvtColor(viz_map, cv.COLOR_GRAY2BGR)
        # for cp in self.critical_points:
        #     y, x = cp
        #     skel_div_map_cp[y, x] = (0, 0, 255)
        # # show error points
        # for ep in self.error_point1:
        #     y, x = ep
        #     skel_div_map_cp[y, x] = (255, 0, 0)  # error points in blue
        # for ep in self.error_point2:
        #     y, x = ep
        #     skel_div_map_cp[y, x] = (255, 255, 0)  # error points in cyan
        # cv.imshow("Division Map with Critical Points", skel_div_map_cp)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
