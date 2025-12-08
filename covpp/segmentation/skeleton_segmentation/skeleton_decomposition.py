"""Voronoi-based segmentation for optimal workspace decomposition."""
import dataclasses
import cv2 as cv
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize  # , medial_axis
from covpp.base.logger import MyLogger
from covpp.base.configs import RobotConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.map_polygons import MapPolygons
# from tsmoothie.smoother import LowessSmoother


@dataclasses.dataclass
class Skelet:
    """ Contains skeletonization results. """
    contour_map: np.ndarray            # dtype=uint8 0, 255
    skeleton: np.ndarray               # dtype=bool
    distance: np.ndarray               # dtype=uint8
    distance_normalized: np.ndarray    # dtype=uint8 0-255
    distance_skeleton: np.ndarray      # dtype=uint8 0-255
    endpoints: np.ndarray              # dtype=uint8 0, 1
    branchpoints: np.ndarray           # dtype=uint8 0, 1
    endpoints_filtered: np.ndarray     # dtype=uint8 0, 1


@dataclasses.dataclass
class Branch:
    """ Represents a skeleton branch/path between endpoints/branchpoints. """
    points: np.ndarray                # shape (L, 2), dtype=int
    start: tuple[int, int]            # (y, x)
    end: tuple[int, int]              # (y, x)
    end_type: str                     # 'endpoint' | 'branchpoint' | 'skip'
    length: int                       # length of the branch (number of points)
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

    def skeleton_decomposition(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        self.create_skeleton()
        self.find_endpoints_branchpoints()
        self.extract_special_branches()
        self.lg.info2(f"Extracted {len(self.branches)} skeleton branches.")
        self.find_critical_points()
        self.lg.info2(f"Found {len(self.critical_points)} critical points in skeleton branches.")
        self.find_base_points()
        self.visualize_skeleton()
        raise NotImplementedError("Skeletonization visualization complete.")

    def create_skeleton(self) -> None:
        """Create skeleton of the workspace and find endpoints and branchpoints."""

        # Compute medial axis (skeleton) and the distance transform
        # skel, distance = medial_axis(self.map_ws, return_distance=True)  # ske: dtype=bool, distance: dtype=float64

        # distance transform and normalization
        distance = cv.distanceTransform(self.map_ws, distanceType=cv.DIST_L2, maskSize=5)  # dtype=float32
        distance = distance.astype(np.uint8)
        distance_normalized = (distance / distance.max() * 255).astype(np.uint8)

        # skeletonization
        skeleton = skeletonize(self.map_ws)  # dtype=bool
        # skeleton_lee = skeletonize(self.map_ws, method='lee')

        # distance_normalized masked by skeleton
        distance_skeleton = np.zeros_like(distance_normalized)  # dtype=uint8
        distance_skeleton[skeleton] = distance_normalized[skeleton]

        # create contour map (contours cells = 255 otherwise 0)
        contours, _ = cv.findContours(self.map_ws, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour_map = np.zeros_like(self.map_ws)
        cv.drawContours(contour_map, contours, -1, 255, thickness=1)

        self.skelet = Skelet(
            skeleton=skeleton,
            distance=distance,
            distance_normalized=distance_normalized,
            distance_skeleton=distance_skeleton,
            contour_map=contour_map,
            endpoints=None,
            branchpoints=None,
            endpoints_filtered=None
        )

    def find_endpoints_branchpoints(self) -> None:
        """ Find endpoints and branchpoints in the skeleton. """

        skeleton = self.skelet.skeleton.astype(np.uint8)  # dtype=uint8

        # Convolution kernel to count neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)
        filtered = cv.filter2D(skeleton, -1, kernel)   # dtype=uint8

        # Identify endpoints and branchpoints based on neighbor counts
        endpoints = np.where(filtered == 11, 1, 0).astype(np.uint8)     # dtype=uint8
        branchpoints = np.where(filtered >= 13, 1, 0).astype(np.uint8)  # dtype=uint8

        # filtering endpoints too close to branchpoints
        mask = np.zeros_like(self.zmd.map_ws, dtype=np.uint8)
        branchpoint_coords = np.column_stack(np.where(branchpoints))
        for y, x in branchpoint_coords:
            radius = 1.2 * self.skelet.distance_normalized[y, x]
            cv.circle(mask, (x, y), int(radius), 1, thickness=-1)  # filled circle
        endpoints_filtered_1 = endpoints * (1 - mask)

        # second method for filtering
        # fildter endpoints with same distance as min_distance in skeleton
        distance_skeleton = self.skelet.distance_skeleton
        dists_sorted = np.sort(distance_skeleton[distance_skeleton > 0])
        dists_sorted = dists_sorted[:np.sum(endpoints)]
        dist_thresh = dists_sorted[-1]
        endpoints_filtered2 = np.where(distance_skeleton > dist_thresh, endpoints, 0).astype(np.uint8)

        # combine both filtering methods (AND)
        endpoints_filtered = endpoints_filtered_1 * endpoints_filtered2

        # skeletonization data
        self.skelet.endpoints = endpoints
        self.skelet.branchpoints = branchpoints
        self.skelet.endpoints_filtered = endpoints_filtered

    def extract_special_branches(self) -> list[Branch]:
        """ Extract branches from a skeleton as paths between endpoints/branchpoints. """
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
            dists = self.skelet.distance_normalized[branch.points[:, 0], branch.points[:, 1]]
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
            if min_dist > max_dist * 0.7:
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
        self.find_base_points_perpendicular_lines()
        # self.find_base_points_circle()

    def find_base_points_perpendicular_lines(self):
        """Find base points for critical points.

        For each critical point, find two base points by extending lines
        in both perpendicular directions to the tangent at the critical point until hitting contour or occupied space.
        """
        base_points_all = []

        # find branch angle at critical points
        for branch in self.branches:
            for cp_idx, cp in zip(branch.cp_idxs, branch.cp_coords):
                # estimate tangent using neighboring (d_idx) points
                d_idx = 5  # param
                if cp_idx == 0:
                    p1 = branch.points[cp_idx]
                    p2 = branch.points[cp_idx + d_idx]
                elif cp_idx == branch.length - 1:
                    p1 = branch.points[cp_idx - d_idx]
                    p2 = branch.points[cp_idx]
                else:
                    idx1 = max(0, cp_idx - d_idx)
                    idx2 = min(branch.length - 1, cp_idx + d_idx)
                    p1 = branch.points[idx1]
                    p2 = branch.points[idx2]

                dy = p2[0] - p1[0]
                dx = p2[1] - p1[1]
                angle = np.arctan2(dy, dx)
                # perpendicular angle
                perp_angle = angle + np.pi / 2

                # search line in both perpendicular directions till hitting contour or occupied space
                height, width = self.map_ws.shape
                dist_cp = int(self.skelet.distance[cp[0], cp[1]])
                r = dist_cp
                base_pts = []
                for sign in [1, -1]:
                    for dist in range(r-10, 2*r):
                        y = cp[0] + int(round(sign * dist * np.sin(perp_angle)))
                        x = cp[1] + int(round(sign * dist * np.cos(perp_angle)))
                        if 0 <= y < height and 0 <= x < width:
                            if self.skelet.contour_map[y, x] == 255 or self.map_ws[y, x] == 0:
                                base_pts.append((y, x))
                                break
                if len(base_pts) == 0:
                    self.lg.error(f"No base points found for critical point at {cp}.")
                    continue
                if len(base_pts) == 1:
                    self.lg.error(f"Only one base point found for critical point at {cp}.")
                base_points_all.append(np.array(base_pts, dtype=int))

        self.base_points = base_points_all

    def find_base_points_circle(self):
        """
        Find base points for critical points.
        For each critical point, sample points on the circle of radius `dist_cp`
        around the critical point and collect those lying on occupied/contour space.

        Incomplete method: more than two base points may be found. 
        (solution: create contours based on boundary and medial axis and create masks for each closed area)
        """
        base_points_all = []

        height, width = self.map_ws.shape
        for cp in self.critical_points:
            y_cp, x_cp = cp
            dist_cp = self.skelet.distance[y_cp, x_cp]
            if dist_cp <= 0:
                self.lg.error(f"Invalid distance at critical point {cp}.")
                continue
            # Sample points on circle around critical point
            angles = np.linspace(0, 2 * np.pi, num=360, endpoint=False)
            pts = []
            r = dist_cp + 1
            for ang in angles:
                y = y_cp + int(round(r * np.sin(ang)))
                x = x_cp + int(round(r * np.cos(ang)))
                if 0 <= y < height and 0 <= x < width:
                    if self.skelet.contour_map[y, x] == 255:
                        pts.append((y, x))

            # If none found on contour, fallback to occupied map border (value 0)
            if not pts:
                for ang in angles:
                    y = y_cp + int(round(r * np.sin(ang)))
                    x = x_cp + int(round(r * np.cos(ang)))
                    if 0 <= y < height and 0 <= x < width:
                        if self.map_ws[y, x] == 0:
                            pts.append((y, x))
            if pts:
                base_points_all.append(np.array(pts, dtype=int))

        self.base_points = base_points_all

    # ==========================================================================
    # Branch extraction method
    # ==========================================================================

    def create_graph_and_specials(self) -> tuple[nx.Graph, set]:
        """ Create graph from skeleton and identify special nodes (endpoints and branchpoints). """

        # Build graph
        graph = nx.Graph()
        height, width = self.skelet.skeleton.shape
        coords = np.column_stack(np.where(self.skelet.skeleton))
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
        ep_coords = np.column_stack(np.where(self.skelet.endpoints_filtered))
        bp_coords = np.column_stack(np.where(self.skelet.branchpoints))
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

    def extract_special_branches_from_graph(self, graph: nx.Graph, special: set) -> list[Branch]:
        """ Extract branches from the graph between special nodes or endpoints. """

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
                                           start=start,
                                           end=end,
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
        distance = self.skelet.distance
        endpoints = self.skelet.endpoints
        endpoints_filtered = self.skelet.endpoints_filtered
        distance_skeleton = self.skelet.distance_skeleton
        branchpoints = self.skelet.branchpoints
        branches = self.branches

        # # visualize using OpenCV
        cv.imshow("distance", distance)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # # visualization using matplotlib
        colors = plt.cm.get_cmap('hsv', len(branches))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.contour(self.zmd.map_ws, levels=[0.5], colors='white', linewidths=0.5)
        ax.imshow(distance_skeleton, cmap='magma', alpha=1.0)
        ax.scatter(*np.where(endpoints == 1)[::-1], color='cyan', s=10, label='Endpoints')
        ax.scatter(*np.where(branchpoints == 1)[::-1], color='red', s=10, label='Branchpoints')
        ax.scatter(*np.where(endpoints_filtered == 1)[::-1], color='lime', s=20, label='Filtered Endpoints')
        # branches
        for i, branch in enumerate(branches):
            yy, xx = branch.points[:, 0], branch.points[:, 1]
            color = 'orange' if branch.end_type == 'skip' else colors(i)
            plt.plot(xx, yy, color=color, linewidth=1)
        # critical points
        critical_points = np.array(self.critical_points)
        ax.scatter(critical_points[:, 1], critical_points[:, 0], color='magenta', s=30, label='Local Min Dists')
        # base points
        for base_pair in self.base_points:
            ax.scatter(base_pair[:, 1], base_pair[:, 0], color='yellow', s=15)

        ax.set_title('Voronoi Skeletonization Results')
        ax.legend()
        ax.axis('off')
        plt.show()
