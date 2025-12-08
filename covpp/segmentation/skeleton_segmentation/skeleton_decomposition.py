"""Voronoi-based segmentation for optimal workspace decomposition."""
import dataclasses
import cv2 as cv
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize  # , medial_axis
from covpp.base.logger import MyLogger
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.map_polygons import MapPolygons
# from covpp.base.configs import PipeLineConfig


@dataclasses.dataclass
class Skelet:
    """ Contains skeletonization results. """
    skeleton: np.ndarray               # dtype=bool
    distance: np.ndarray               # dtype=float32
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
    length: int


class SkeletonDecomposition:
    """Skeletonization-based decomposition of the zone map."""

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        self.lg = MyLogger("SkeletonDecomposition")
        self.lg.info2("Initializing Skeleton Decomposition.")
        self.zmd = zmd
        self.map_polygons = map_polygons

        self.skelet: Skelet = None                  # skeletonization data
        self.branches: list[Branch] = []            # skeleton branches
        self.critiacl_points: list[tuple[int, int]] = []     # local min distance points in branches
        self.base_points = []                     # base points for critical points

    def skeleton_decomposition(self) -> None:
        """Run skeletonization-based segmentation pipeline."""

        self.create_skeleton()
        self.find_endpoints_branchpoints()
        self.extract_special_branches()
        self.find_critical_points()
        self.find_base_points()
        self.visualize_skeleton()
        raise NotImplementedError("Skeletonization visualization complete.")

    def create_skeleton(self) -> None:
        """Create skeleton of the workspace and find endpoints and branchpoints."""

        # input map blobs
        blobs = self.zmd.map_cleaning  # uint8 binary map

        # Compute medial axis (skeleton) and the distance transform
        # skel, distance = medial_axis(blobs, return_distance=True)  # ske: dtype=bool, distance: dtype=float64

        # distance transform and normalization
        distance = cv.distanceTransform(blobs, distanceType=cv.DIST_L2, maskSize=5)  # dtype=float32
        distance_normalized = (distance / distance.max() * 255).astype(np.uint8)

        # skeletonization
        skeleton = skeletonize(blobs)  # dtype=bool
        # skeleton_lee = skeletonize(blobs, method='lee')

        # distance_normalized masked by skeleton
        distance_skeleton = np.zeros_like(distance_normalized)  # dtype=uint8
        distance_skeleton[skeleton] = distance_normalized[skeleton]

        self.skelet = Skelet(
            skeleton=skeleton,
            distance=distance_normalized,
            distance_skeleton=distance_skeleton,
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
            radius = 1.2 * self.skelet.distance[y, x]
            cv.circle(mask, (x, y), int(radius), 1, thickness=-1)  # filled circle
        endpoints_filtered_1 = endpoints * (1 - mask)

        # second method for filtering
        # fildter endpoints with same distance as min_distance in skeleton
        distance_skeleton = self.skelet.distance_skeleton
        dists_sorted = np.sort(distance_skeleton[distance_skeleton > 0])
        dists_sorted = dists_sorted[:np.sum(endpoints)]
        dist_thresh = dists_sorted[-1]
        print(f"Min distance in skeleton: {dists_sorted}")
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

    def find_critical_points(self):
        """ Find local minimum distance points in each branch.
        If there are multiple points (next to each other) in the branch with the same minimum distance, select the middle one.
        """
        local_points = []
        for branch in self.branches:
            branch_dists = self.skelet.distance[branch.points[:, 0], branch.points[:, 1]]
            max_dist = branch_dists.max()
            min_dist = branch_dists.min()
            min_indices = np.where(branch_dists == min_dist)[0]
            # # select middle index if multiple
            # if len(min_indices) > 1:
            #     mid_idx = min_indices[len(min_indices) // 2]
            # else:
            #     mid_idx = min_indices[0]

            # only consider if min_dist is significantly less than max_dist
            if min_dist > max_dist * 0.7:
                continue

            local_ps = branch.points[min_indices]
            local_points.extend(local_ps)
        self.critiacl_points = local_points

    def find_base_points(self):
        """
        Find base points for critical points.
        For each critical point, find the two base points which are the closest points on the occupied map space.

        Steps:
        - For each critical point, find direction perpendicular to the skeleton at that point.
        - Move along that direction in both ways until hitting occupied space.
        """

        base_points_all = []
        for cp in self.critiacl_points:
            y_cp, x_cp = cp
            dist_cp = self.skelet.distance[y_cp, x_cp]
            # search in a circle around the critical point
            angles = np.linspace(0, 2 * np.pi, num=36, endpoint=False)
            base_pts_cp = []
            for angle in angles:
                for r in range(int(dist_cp), int(dist_cp * 3)):
                    y_off = int(round(r * np.sin(angle)))
                    x_off = int(round(r * np.cos(angle)))
                    y_search = y_cp + y_off
                    x_search = x_cp + x_off
                    # check bounds
                    if (0 <= y_search < self.zmd.map_ws.shape[0]) and (0 <= x_search < self.zmd.map_ws.shape[1]):
                        # check if occupied
                        if self.zmd.map_ws[y_search, x_search] == 0:
                            base_pts_cp.append((y_search, x_search))
                            break  # move to next angle after finding first occupied point
            base_points_all.append(base_pts_cp)
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

                # collect branch info
                if end_type != 'skip':
                    branches.append(Branch(points=points,
                                           start=start,
                                           end=end,
                                           end_type=end_type,
                                           length=len(points),
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
        endpoints = self.skelet.endpoints
        endpoints_filtered = self.skelet.endpoints_filtered
        distance_skeleton = self.skelet.distance_skeleton
        branchpoints = self.skelet.branchpoints
        branches = self.branches

        # # # visualize using OpenCV
        # cv.imshow("distance_skeleton", distance_skeleton)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

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
        critiacl_points = np.array(self.critiacl_points)
        ax.scatter(critiacl_points[:, 1], critiacl_points[:, 0], color='magenta', s=30, label='Local Min Dists')
        # base points
        for base_pts_cp in self.base_points:
            base_pts_cp = np.array(base_pts_cp)
            ax.scatter(base_pts_cp[:, 1], base_pts_cp[:, 0], color='yellow', s=10, label='Base Points')

        ax.set_title('Voronoi Skeletonization Results')
        ax.legend()
        ax.axis('off')
        plt.show()
