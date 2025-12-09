"""Voronoi-based segmentation for optimal workspace decomposition."""
import dataclasses
import heapq
import cv2 as cv
import numpy as np
import networkx
from matplotlib import pyplot as plt
from skimage.morphology import medial_axis  # , skeletonize
from covpp.base.logger import MyLogger
from covpp.base.configs import RobotConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.map_polygons import MapPolygons
# from tsmoothie.smoother import LowessSmoother


@dataclasses.dataclass
class Skelet:
    """ Contains skeletonization results. """
    contour_map: np.ndarray            # dtype=uint8 0, 255
    skeleton_map: np.ndarray               # dtype=bool
    distance_map: np.ndarray               # dtype=uint8
    distance_norm_map: np.ndarray    # dtype=uint8 0-255
    distance_skeleton_map: np.ndarray      # dtype=uint8 0-255
    end_points: np.ndarray              # dtype=uint8 0, 1
    branch_points: np.ndarray           # dtype=uint8 0, 1
    end_points_filtered: np.ndarray     # dtype=uint8 0, 1


@dataclasses.dataclass
class Branch:
    """ Represents a skeleton branch/path between end_points/branch_points. """
    length: int                             # length of the branch (number of points)
    end_type: str                           # 'endpoint' | 'branchpoint' | 'skip'
    points: np.ndarray                      # shape (L, 2), dtype=int
    end_point: tuple[int, int]              # (y, x)
    start_point: tuple[int, int]            # (y, x)
    cp_idxs: list[int] = dataclasses.field(default_factory=list)                # indices in critical points list
    cp_coords: list[tuple[int, int]] = dataclasses.field(default_factory=list)  # list of critical point (y, x) coordinates in this branch


class SkeletonDecomposition:
    """Skeletonization-based decomposition of the zone map."""

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        self.lg = MyLogger("SkeletonDecomposition")
        self.lg.info2("Initializing Skeleton Decomposition.")
        self.zmd = zmd
        self.map_polygons = map_polygons
        self.map_ws = zmd.map_ws  # uint8 binary map (0, 255)
        self.robot_radius_px = int(RobotConfig.radius / zmd.resolution)

        self.skelet: Skelet = None                  # skeletonization data
        self.branches: list[Branch] = []            # skeleton branches
        self.critical_points: list[tuple[int, int]] = []     # local min distance points in branches
        self.base_points = []                     # base points for critical points
        self.side_points = []                     # side points for critical points
        self.expandion_points = []               # expansion points during base point search

    def skeleton_decomposition(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        self.create_skeleton()
        self.find_endpoints_branchpoints()
        self.extract_special_branches()
        self.lg.info2(f"Extracted {len(self.branches)} skeleton branches.")
        self.find_critical_points()
        self.lg.info2(f"Found {len(self.critical_points)} critical points in skeleton branches.")
        self.find_base_points()
        self.lg.info2(f"Found {len(self.base_points)} base points for critical points.")
        self.visualize_skeleton()
        raise NotImplementedError("Skeletonization visualization complete.")

    def create_skeleton(self) -> None:
        """Create skeleton of the workspace and find end_points and branch_points."""

        # Compute medial axis (skeleton) and the distance transform
        skeleton_map, distance_map = medial_axis(self.map_ws, return_distance=True)  # ske: dtype=bool, distance: dtype=float64

        # distance transform and normalization
        # distance_map = cv.distanceTransform(self.map_ws, distanceType=cv.DIST_L2, maskSize=5)  # dtype=float32
        distance_map = distance_map.astype(np.uint8)
        distance_norm_map = (distance_map / distance_map.max() * 255).astype(np.uint8)

        # # skeletonization
        # skeleton_map = skeletonize(self.map_ws)  # dtype=bool
        # skeleton_lee = skeletonize(self.map_ws, method='lee')

        # distance_normalized masked by skeleton
        distance_skeleton_map = np.zeros_like(distance_norm_map)  # dtype=uint8
        distance_skeleton_map[skeleton_map] = distance_norm_map[skeleton_map]

        # create contour map (contours cells = 255 otherwise 0)
        contours, _ = cv.findContours(self.map_ws, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_map = np.zeros_like(self.map_ws)
        cv.drawContours(contour_map, contours, -1, 255, thickness=1)

        self.skelet = Skelet(
            skeleton_map=skeleton_map,
            distance_map=distance_map,
            distance_norm_map=distance_norm_map,
            distance_skeleton_map=distance_skeleton_map,
            contour_map=contour_map,
            end_points=None,
            branch_points=None,
            end_points_filtered=None
        )

    def find_endpoints_branchpoints(self) -> None:
        """ Find end_points and branch_points in the skeleton. """

        skeleton = self.skelet.skeleton_map.astype(np.uint8)  # dtype=uint8

        # Convolution kernel to count neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)
        filtered = cv.filter2D(skeleton, -1, kernel)   # dtype=uint8

        # Identify end_points and branch_points based on neighbor counts
        end_points = np.where(filtered == 11, 1, 0).astype(np.uint8)     # dtype=uint8
        branch_points = np.where(filtered >= 13, 1, 0).astype(np.uint8)  # dtype=uint8

        # filtering end_points too close to branch_points
        mask = np.zeros_like(self.zmd.map_ws, dtype=np.uint8)
        branchpoint_coords = np.column_stack(np.where(branch_points))
        for y, x in branchpoint_coords:
            radius = 1.2 * self.skelet.distance_norm_map[y, x]
            cv.circle(mask, (x, y), int(radius), 1, thickness=-1)  # filled circle
        end_points_filtered_1 = end_points * (1 - mask)

        # second method for filtering
        # fildter end_points with same distance as min_distance in skeleton
        distance_skeleton = self.skelet.distance_skeleton_map
        dists_sorted = np.sort(distance_skeleton[distance_skeleton > 0])
        dists_sorted = dists_sorted[:np.sum(end_points)]
        dist_thresh = dists_sorted[-1]
        end_points_filtered2 = np.where(distance_skeleton > dist_thresh, end_points, 0).astype(np.uint8)

        # combine both filtering methods (AND)
        end_points_filtered = end_points_filtered_1 * end_points_filtered2

        # skeletonization data
        self.skelet.end_points = end_points
        self.skelet.branch_points = branch_points
        self.skelet.end_points_filtered = end_points_filtered

    def extract_special_branches(self) -> list[Branch]:
        """ Extract branches from a skeleton as paths between end_points/branch_points. """
        graph, special = self.create_graph_and_specials()
        self.branches = self.extract_special_branches_from_graph(graph, special)

    # ==========================================================================
    # find critical points and base points
    # ==========================================================================

    def find_critical_points(self):
        """ Find local minimum distance points in each branch.
        If there are multiple points (next to each other) in the branch with the same minimum distance, select the middle one.
        """
        cp_points = []  # list of (y, x) critical points
        for branch in self.branches:
            dists = self.skelet.distance_norm_map[branch.points[:, 0], branch.points[:, 1]]
            # smoother = LowessSmoother(smooth_fraction=0.2, iterations=1)
            # smoother.smooth(dists.tolist())
            # dists_smooth = np.array(smoother.smooth_data[0])

            # evaluate distances from branch ends and overall min/max
            min_end_dist = min(dists[0], dists[-1])
            max_dist = dists.max()
            min_dist = dists.min()
            d_min_max = max_dist - min_dist
            d_min_end = min_end_dist - min_dist
            if d_min_end < 0.1 * d_min_max:
                # skip if min distance is close to end distances
                continue

            min_indices = np.where(dists == min_dist)[0]
            # # select middle index if multiple
            # if len(min_indices) > 1:
            #     mid_idx = min_indices[len(min_indices) // 2]
            # else:
            #     mid_idx = min_indices[0]

            # only consider if min_dist is significantly less than max_dist
            if min_dist > max_dist * 0.8:
                continue

            # take the first one and last one (to avoid biasing to one end)
            min_indices = np.array([min_indices[0], min_indices[-1]])
            local_ps = branch.points[min_indices]
            cp_points.extend(local_ps)

            # update branch info
            branch.cp_coords = local_ps.tolist()
            branch.cp_idxs = min_indices.tolist()

        # update critical points list
        self.critical_points = cp_points

    def find_base_points(self):
        """ Find base points for critical points. """
        # self.find_base_points_perpendicular_lines()
        # self.find_base_points_circle()
        self.find_base_points_by_side_expansion()

    def find_base_points_by_side_expansion(self):
        """Find base points for critical points."""

        for branch in self.branches:
            for cp in branch.cp_coords:
                base_pts = self.get_base_points_by_find_expand_side_points(cp)
                if base_pts is None:
                    self.lg.error(f"No base points found for critical point at {cp}.")
                if len(base_pts) != 2:
                    self.lg.error(f"Expected 2 base points, found {len(base_pts)} for critical point at {cp}.")
                self.base_points.extend(base_pts)

    def get_base_points_by_find_expand_side_points(self, cp) -> np.ndarray:
        """ Find two side points for a critical point and expand until finding base points.

        Expansion is guided by `distance_map`: neighbors with lower distance are explored first,
        so the search tends toward walls/contours faster.
        """

        # dxy_4c = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dxy_8c = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)]

        # find neighbors (4-connectivity) of cp
        neighbor_points: list[np.ndarray] = []
        for dy, dx in dxy_8c:
            y, x = cp
            y += dy
            x += dx
            # ingore if on skeleton
            if self.skelet.skeleton_map[y, x]:
                continue
            neighbor_points.append(np.array((y, x), dtype=int))
        if len(neighbor_points) < 2:
            self.lg.error(f"Could not find two neighbor points for critical point at {cp}.")
            return None

        # select two neighbors that are farthest apart
        max_dist = -1
        side_points: list[np.ndarray] = []
        for point_i in neighbor_points:
            for point_j in neighbor_points:
                dist = np.linalg.norm(point_i - point_j)
                if dist > max_dist:
                    max_dist = dist
                    side_points = [point_i, point_j]
        self.side_points.extend(side_points)

        # expand like graph search two side points until hitting contour or occupied space (stop expansion if reached skeleton)
        base_pts = []
        for sp in side_points:
            visited: set[tuple[int, int]] = set()
            heap: list[tuple[int, int, int, int]] = []  # (priority=distance, steps, y, x)
            sy, sx = sp[0], sp[1]
            heapq.heappush(heap, (self.skelet.distance_norm_map[sy, sx], 0, sy, sx))

            found_base = None
            while heap and found_base is None:
                _, steps, cy, cx = heapq.heappop(heap)
                if (cy, cx) in visited:
                    continue
                visited.add((cy, cx))
                self.expandion_points.append((cy, cx))

                # check if hitting contour or occupied space
                if self.skelet.contour_map[cy, cx] == 255 or self.map_ws[cy, cx] == 0:
                    found_base = (cy, cx)
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

            if found_base is not None:
                base_pts.append(np.array(found_base, dtype=int))
        print(base_pts)
        return np.array(base_pts, dtype=int)

    # ==========================================================================
    # Branch extraction method
    # ==========================================================================

    def create_graph_and_specials(self) -> tuple[networkx.Graph, set]:
        """ Create graph from skeleton and identify special nodes (end_points and branch_points). """

        # Build graph
        graph = networkx.Graph()
        height, width = self.skelet.skeleton_map.shape
        coords = np.column_stack(np.where(self.skelet.skeleton_map))
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
        special = set()
        ep_coords = np.column_stack(np.where(self.skelet.end_points_filtered))
        bp_coords = np.column_stack(np.where(self.skelet.branch_points))
        endpoint_set = set()
        branchpoint_set = set()
        for y, x in ep_coords:
            if (y, x) in graph:
                node = (y, x)
                endpoint_set.add(node)
                special.add(node)
        for y, x in bp_coords:
            if (y, x) in graph:
                node = (y, x)
                branchpoint_set.add(node)
                special.add(node)
        # # Fallback: classify by degree if masks are imperfect
        # for n in list(graph.nodes):
        #     deg = graph.degree[n]
        #     if deg == 1 or deg >= 3:
        #         special.add(n)

        return graph, special

    def extract_special_branches_from_graph(self, graph: networkx.Graph, special: set) -> list[Branch]:
        """ Extract branches from the graph between special nodes or end_points. """

        # Traverse branches:
        # from each special node along degree-2 nodes to next special or endpoint
        visited_edges = set()
        branches = []

        # Start walks from each special node
        for s in special:
            for v in graph.neighbors(s):
                e = tuple(sorted((s, v)))
                if e in visited_edges:
                    continue
                # Walk until next special/junction
                path = self.walk_branch(s, v, special, graph)
                # Mark all edges in the path as visited
                for a, b in zip(path[:-1], path[1:]):
                    visited_edges.add(tuple(sorted((a, b))))
                start = path[0]
                end = path[-1]
                points = np.array(path, dtype=int)

                # Classify end type
                if end in special:
                    end_type = 'special'
                else:
                    end_type = 'skip'
                # deg_end = graph.degree[end]
                # if deg_end == 1:
                #     end_type = 'endpoint-by-degree'
                # elif deg_end >= 3:
                #     end_type = 'junction'
                # else:
                #     end_type = 'dangling'

                # check branch length
                len_branch = len(points)
                if len_branch < 4 * self.robot_radius_px:
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

    def walk_branch(self, start, nxt, special, graph) -> list[tuple[int, int]]:
        """ Walk along a branch from start to next special node or endpoint."""
        path = [start, nxt]
        prev = start
        curr = nxt
        while True:
            # deg = graph.degree[curr]
            if curr in special and curr != start:
                # Hit terminal special node
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

    # ==========================================================================
    # visualization methods
    # ==========================================================================

    def visualize_skeleton(self) -> None:
        """Visualize skeletonization results."""

        # data
        # distance = self.skelet.distance_map
        branches = self.branches
        end_points = self.skelet.end_points
        branch_points = self.skelet.branch_points
        end_points_filtered = self.skelet.end_points_filtered
        distance_skeleton = self.skelet.distance_skeleton_map
        # distance_norm_map = self.skelet.distance_norm_map
        # distance_map = self.skelet.distance_map

        # # # visualize using OpenCV
        # cv.imshow("distance", distance)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # # visualization using matplotlib
        colors = plt.cm.get_cmap('hsv', len(branches))
        _, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.contour(self.zmd.map_ws, levels=[0.5], colors='white', linewidths=0.5)
        ax.imshow(distance_skeleton, cmap='magma', alpha=1.0)
        # ax.imshow(distance_map, cmap='magma', alpha=1.0)
        ax.scatter(*np.where(end_points == 1)[::-1], color='cyan', s=10, label='Endpoints')
        ax.scatter(*np.where(branch_points == 1)[::-1], color='red', s=10, label='Branchpoints')
        ax.scatter(*np.where(end_points_filtered == 1)[::-1], color='lime', s=20, label='Filtered Endpoints')
        # branches
        for i, branch in enumerate(branches):
            yy, xx = branch.points[:, 0], branch.points[:, 1]
            color = 'orange' if branch.end_type == 'skip' else colors(i)
            plt.plot(xx, yy, color=color, linewidth=1)
        # critical points
        critical_points = np.array(self.critical_points)
        ax.scatter(critical_points[:, 1], critical_points[:, 0], color='magenta', s=30, label='Local Min Dists')
        # base points
        for bp in self.base_points:
            ax.scatter(bp[1], bp[0], color='yellow', s=15)
        # side points
        side_points = np.array(self.side_points)
        ax.scatter(side_points[:, 1], side_points[:, 0], color='cyan', s=10)
        # expansion points
        expand_points = np.array(self.expandion_points)
        ax.scatter(expand_points[:, 1], expand_points[:, 0], color='blue', s=1)

        ax.set_title('Voronoi Skeletonization Results')
        ax.legend()
        ax.axis('off')
        plt.show()


# ==========================================================================
# find base points
# ==========================================================================

# def find_base_points_perpendicular_lines(self):
#     """Find base points for critical points.

#     For each critical point, find two base points by extending lines
#     in both perpendicular directions to the tangent at the critical point until hitting contour or occupied space.
#     """
#     base_points_all = []

#     # find branch angle at critical points
#     for branch in self.branches:
#         for cp_idx, cp in zip(branch.cp_idxs, branch.cp_coords):
#             # estimate tangent using neighboring (d_idx) points
#             d_idx = 5  # paramz
#             idx1 = max(0, cp_idx - d_idx)
#             idx2 = min(branch.length - 1, cp_idx + d_idx)
#             p1 = branch.points[idx1]
#             p2 = branch.points[idx2]
#             # tangent angle
#             dy = p2[0] - p1[0]
#             dx = p2[1] - p1[1]
#             angle = np.arctan2(dy, dx)
#             # perpendicular angle
#             perp_angle = angle + np.pi / 2

#             # search line in both perpendicular directions till hitting contour or occupied space
#             height, width = self.map_ws.shape
#             dist_cp = int(self.skelet.distance_map[cp[0], cp[1]])
#             r = dist_cp
#             base_pts = []
#             for sign in [1, -1]:
#                 for dist in range(r-10, 5*r):
#                     y = cp[0] + int(round(sign * dist * np.sin(perp_angle)))
#                     x = cp[1] + int(round(sign * dist * np.cos(perp_angle)))
#                     if 0 <= y < height and 0 <= x < width:
#                         if self.skelet.contour_map[y, x] == 255 or self.map_ws[y, x] == 0:
#                             base_pts.append((y, x))
#                             break
#             if len(base_pts) == 0:
#                 self.lg.error(f"No base points found for critical point at {cp}.")
#                 continue
#             if len(base_pts) == 1:
#                 self.lg.error(f"Only one base point found for critical point at {cp}.")
#             base_points_all.append(np.array(base_pts, dtype=int))

#     self.base_points = base_points_all

# def find_base_points_circle(self):
#     """
#     Find base points for critical points.
#     For each critical point, sample points on the circle of radius `dist_cp`
#     around the critical point and collect those lying on occupied/contour space.

#     Incomplete method: more than two base points may be found.
#     (solution: create contours based on boundary and medial axis and create masks for each closed area)
#     """
#     base_points_all = []

#     height, width = self.map_ws.shape
#     for cp in self.critical_points:
#         y_cp, x_cp = cp
#         dist_cp = self.skelet.distance_map[y_cp, x_cp]
#         if dist_cp <= 0:
#             self.lg.error(f"Invalid distance at critical point {cp}.")
#             continue
#         # Sample points on circle around critical point
#         angles = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
#         pts = []
#         r = dist_cp + 1
#         for ang in angles:
#             y = y_cp + int(round(r * np.sin(ang)))
#             x = x_cp + int(round(r * np.cos(ang)))
#             if 0 <= y < height and 0 <= x < width:
#                 if self.skelet.contour_map[y, x] == 255:
#                     pts.append((y, x))

#         # If none found on contour, fallback to occupied map border (value 0)
#         if not pts:
#             for ang in angles:
#                 y = y_cp + int(round(r * np.sin(ang)))
#                 x = x_cp + int(round(r * np.cos(ang)))
#                 if 0 <= y < height and 0 <= x < width:
#                     if self.map_ws[y, x] == 0:
#                         pts.append((y, x))
#         if pts:
#             base_points_all.append(np.array(pts, dtype=int))

#     self.base_points = base_points_all
