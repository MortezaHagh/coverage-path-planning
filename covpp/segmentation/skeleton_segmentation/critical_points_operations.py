
import heapq
import numpy as np
from covpp.base.logger import MyLogger
from covpp.segmentation.skeleton_segmentation.data_structures import Branch, CriticalPoint, Configs


class CriticalPointOperations:
    """ Class for finding critical points and base points in the skeleton of a map. """

    def __init__(self, distance_map: np.ndarray, skeleton_map_b: np.ndarray, branches: list[Branch]):
        self.lg = MyLogger("CriticalPointOperations")
        self.cfgs = Configs()
        self.distance_map: np.ndarray = distance_map
        self.skeleton_map_b: np.ndarray = skeleton_map_b
        self.branches: list[Branch] = branches

        # attributes
        self.critical_pts: list[CriticalPoint] = []
        self.base_pts: list[tuple[int, int]] = []

    def run(self):
        self.find_critical_points()
        self.find_base_points()

    def find_critical_points(self):
        """ Find local minimum distance points in each branch.
        If there are multiple points (next to each other) in the branch with the same minimum distance, select the middle one.
        """
        cp_pts: list[np.ndarray] = []  # list of (y, x) critical points
        cps: list[CriticalPoint] = []
        for branch in self.branches:
            dists = self.distance_map[branch.points[:, 0], branch.points[:, 1]]
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
            if self.cfgs.end_dist_threshold_ratio > 0:
                min_end_dist = min(dists[0], dists[-1])
                d_min_end = min_end_dist - min_dist
                if d_min_end < self.cfgs.end_dist_threshold_ratio * dist_del:
                    continue

            # evaluate length of min_indices
            min_dist_thresh = min_dist + 0.1 * dist_del
            min_indices = np.where(dists < min_dist_thresh)[0]
            if len(min_indices) / branch.length > self.cfgs.max_critical_region_ratio:
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
            if min_dist > max_dist * self.cfgs.min_max_distance_ratio:
                continue

            # select middle index if multiple
            mid_idx = min_indices[len(min_indices) // 2]
            # if len(min_indices) > 1:
            #     mid_idx = min_indices[len(min_indices) // 2]
            # else:
            #     mid_idx = min_indices[0]

            # min_indices = np.array([min_indices[0]])
            min_indices = np.array([mid_idx])
            local_ps = branch.points[min_indices]
            cp_pts.extend(local_ps)
            cps.extend([CriticalPoint(coord=lp) for lp in local_ps])

        # update critical points list
        self.critical_pts.extend(cps)

    def find_base_points(self):
        """Find base points for all critical points."""

        for cp in self.critical_pts:
            self.get_base_points_by_expand_side_points(cp)

        # validate
        if len(self.base_pts) != len(self.critical_pts):
            self.lg.warn(f"N base points {len(self.base_pts)} != N critical points {len(self.critical_pts)}.")
            # raise ValueError("Base points and critical points count mismatch.")

    def find_side_points_simple(self, cp: CriticalPoint, n_pts: int = 2) -> bool:
        ''' Find side points for a critical point using a simple method.'''

        # n = 8
        dxys = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        cp_coord = cp.coord
        adj_idxs = []
        for i, dxy in enumerate(dxys):
            y, x = cp_coord[0] + dxy[0], cp_coord[1] + dxy[1]
            if self.is_on_skeleton((y, x)):
                adj_idxs.append(i)
        if len(adj_idxs) != 2:
            self.lg.warn(f"Expected 2 neighbors, found {len(adj_idxs)} for critical point at {cp_coord}.")
            return False

        ind1 = (adj_idxs[0] + 1) % 8
        ind2 = adj_idxs[0] - 1
        x1, y1 = cp_coord[1] + dxys[ind1][1], cp_coord[0] + dxys[ind1][0]
        x2, y2 = cp_coord[1] + dxys[ind2][1], cp_coord[0] + dxys[ind2][0]
        if self.is_on_skeleton((y1, x1)) or self.is_on_skeleton((y2, x2)):
            self.lg.warn(f"Expected side points not on skeleton for critical point at {cp_coord}.")
            return False
        side_pts = [(y1, x1), (y2, x2)]
        cp.side_points = side_pts
        return True

    def get_base_points_by_expand_side_points(self, cp: CriticalPoint, n_pts: int = 2) -> bool:
        """ Find two side points for a critical point and expand until finding base points.

        Expansion is guided by `distance_map`: neighbors with lower distance are explored first,
        so the search tends toward walls/contours faster.
        """

        # find side points
        if not self.find_side_points_simple(cp, n_pts=n_pts):
            return False

        # expand like graph search side points until hitting contour or occupied space (stop expansion if reached skeleton)
        visited: set[tuple] = set()
        base_pts: list[tuple[int, int]] = []            # base points
        base_paths: list[tuple[tuple[int, int]]] = []   # base paths
        dxy_8c = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # for each side point, expand until hitting contour or occupied space
        # for side_pt, sk_segment in zip(cp.side_points, cp.side_sk_segments):
        for side_pt in cp.side_points:
            base_path = []
            heap: list[tuple[int, int, int, int]] = []  # (priority=distance, steps, y, x)
            sy, sx = side_pt[0], side_pt[1]
            heapq.heappush(heap, (self.distance_map[sy, sx], 0, sy, sx))

            base_p = None
            while heap and base_p is None:
                _, steps, cy, cx = heapq.heappop(heap)
                if (cy, cx) in visited:
                    continue
                visited.add((cy, cx))
                base_path.append((cy, cx))

                # check if hitting contour or occupied space
                if self.distance_map[cy, cx] == 0:
                    base_p = (cy, cx)
                    break

                # if on skeleton, do not expand further
                if self.skeleton_map_b[cy, cx]:
                    continue

                # # check if in the same skeleton segment
                # sk_seg = self.skelet.skeleton_segmented_map[cy, cx]
                # if sk_seg != sk_segment:
                #     continue

                # expand to neighbors prioritized by lower distance
                for ddy, ddx in dxy_8c:
                    ny, nx = cy + ddy, cx + ddx
                    if (ny, nx) in visited:
                        continue
                    # nprio = self.distance_map[ny, nx]
                    nprio = self.distance_map[ny, nx]
                    new_steps = steps + 1
                    if ddy != 0 and ddx != 0:
                        new_steps += 1  # diagonal penalty
                    heapq.heappush(heap, (nprio, new_steps, ny, nx))

            if base_p is not None:
                base_pts.append(base_p)
                base_paths.append(tuple(base_path))

        # validate results
        if len(base_pts) == 0:
            self.lg.error(f"No base points found for critical point at {cp.coord}.")
            return False
        if len(base_pts) != n_pts:
            self.lg.error(f"Expected {n_pts} base points, found {len(base_pts)} for critical point at {cp.coord}.")
            return False

        # store results
        cp.base_pts = base_pts
        cp.base_paths = base_paths
        self.base_pts.append(tuple(base_pts))
        return True

    # ==================================================================================================================
    # utility methods
    # ==================================================================================================================

    def is_on_skeleton(self, point: tuple[int, int]) -> bool:
        """Check if a point is on the skeleton."""
        y, x = point
        return self.skeleton_map_b[y, x] > 0
