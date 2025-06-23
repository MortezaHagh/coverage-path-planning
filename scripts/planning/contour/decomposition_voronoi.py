from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from shapely import Polygon, Point
# from shapely.affinity import scale
# from scipy import ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d  # pylint: disable=no-name-in-module
from scripts.base.logger import MyLogger
from scripts.base.configs import PlanningConfigs
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.map_polygons import MapPolygons
from scripts.planning.contour.common import VPoint, CriticalPoint


class VoronoiSegmentation:
    def __init__(self, zmd: ZoneMapData, map_polygs: MapPolygons):
        self.ns = "VoronoiSegmentation"
        self.lg = MyLogger(self.ns)
        self.lg.info1("Step: Voronoi decomposition.")

        # settings
        self.pcfgs: PlanningConfigs = PlanningConfigs()
        self.min_poly_points_dist = 100.0
        self.adj_dist_lb: float = 50.0  # distance threshold for adjacent Voronoi vertices
        self.fixed_min_n_adj: int = 4  # fixed minimum number of adjacent Voronoi vertices
        self.fixed_max_n_adj: int = 12  # maximum number of adjacent Voronoi vertices
        self.d_clearance_thresh_p: float = 0.05  # percentage of the clearance difference in finding critical points
        self.robot_r_px: int = self.pcfgs.spa.robot_radius / zmd.reslolution
        # computed
        self.mean_wh: float = None
        self.min_n_adjacents: int = None  # minimum number of adjacent Voronoi vertices

        # logging
        self.lg.debug(f"Robot radius in pixels: {self.robot_r_px:.2f}")

        # basic data
        self.zmd: ZoneMapData = zmd
        self.b_poly: Polygon = None  # bounding polygon
        self.obs_polys: list[Polygon] = []  # obstacle polygons
        self.b_poly = map_polygs.b_poly
        self.obs_polys = map_polygs.obs_polys
        # self.map_polygs: MapPolygons = map_polygs  # store the original polygons for reference

        # data for Voronoi decomposition
        self.vor: Voronoi = None
        self.vps_free = None  # vertices inside the bounding polygon
        self.v_points: list[VPoint] = []  # Voronoi vertices as VPoint objects
        self.critical_idxs: list[int] = []  # indices of critical Voronoi vertices
        self.critical_points: list[CriticalPoint] = []  # critical points

    def reset(self):
        self.vor = None
        self.vps_free = None
        self.v_points = []
        self.critical_idxs = []

    def voronoi_decomposition(self) -> None:
        self.process_polygons()
        self.find_voronoi()
        self.find_free_vertices()
        self.find_basis_points()
        #
        self.find_min_n_adjacents()
        adj_dist_ub = self.calculate_adj_dist_ub()
        self.find_adjacents_by_distance(adj_dist_ub=adj_dist_ub)
        self.find_critical_points()
        self.create_critical_points_list()
        #
        self.plot_voronoi()

    def process_polygons(self):
        b_poly = self.process_polygon(self.b_poly)
        if b_poly is None:
            raise ValueError("Bounding polygon processing failed.")
        self.b_poly = b_poly
        obs_polys = []
        for poly in self.obs_polys:
            new_poly = self.process_polygon(poly, reverse=True)
            if new_poly is None:
                raise ValueError("Obstacle polygon processing failed.")
            obs_polys.append(new_poly)
        self.obs_polys = obs_polys

    def find_voronoi(self):
        self.lg.info("Finding Voronoi diagram.")
        p: Polygon = self.b_poly
        points = p.exterior.coords[:-1]
        for obs_poly in self.obs_polys:
            points += obs_poly.exterior.coords[:-1]
        self.vor = Voronoi(points)
        if self.vor is None:
            raise ValueError("Getting Voronoi failed!")

    def find_free_vertices(self):
        vertices = self.vor.vertices

        bd_poly = self.b_poly.convex_hull
        poly_o1 = bd_poly
        # poly_o1 = scale(bd_poly, xfact=1.2, yfact=1.2, origin='center')

        # # create Voronoi vertices
        # check inside polygons
        for i, vertex in enumerate(vertices):
            self.v_points.append(VPoint(vertex, i))

            if self.b_poly.contains(Point(vertex)):
                self.v_points[-1].free = True

            if poly_o1.contains(Point(vertex)):
                self.v_points[-1].in_poly_o1 = True

        # outside obstacles
        verts_out_obst = []
        for poly in self.obs_polys:
            for vp in self.v_points:
                if vp.free:
                    if not poly.contains(Point(vp.v)):
                        verts_out_obst.append(vp.v)
                    else:
                        vp.free = False
        self.vps_free = np.array(verts_out_obst)

    def find_basis_points(self):
        self.lg.info("Finding basis points for Voronoi vertices.")
        vor = self.vor
        vps = self.v_points
        # vertex_to_sites = defaultdict(set)
        for v_idxs in vor.ridge_vertices:
            if -1 not in v_idxs:
                if vps[v_idxs[0]].free and vps[v_idxs[1]].free:
                    vps[v_idxs[0]].n_feas_ridges += 1
                    vps[v_idxs[0]].ridge_vps.append(vps[v_idxs[1]])
                    vps[v_idxs[1]].n_feas_ridges += 1
                    vps[v_idxs[1]].ridge_vps.append(vps[v_idxs[0]])
            else:
                if v_idxs[0] == -1:
                    idx = v_idxs[1]
                else:
                    idx = v_idxs[0]
                self.v_points[idx].has_inf_adj = True

        #
        for (point_indices, v_idxs) in zip(vor.ridge_points, vor.ridge_vertices):
            if vps[v_idxs[0]].n_feas_ridges >= 1 and vps[v_idxs[1]].n_feas_ridges >= 1:  # todo eval
                vps[v_idxs[0]].set_basis_points(np.array(point_indices), vor.points)
                vps[v_idxs[1]].set_basis_points(np.array(point_indices), vor.points)
                if vps[v_idxs[0]].n_feas_ridges >= 2 and vps[v_idxs[1]].n_feas_ridges >= 2:
                    vps[v_idxs[0]].feasible = True
                    vps[v_idxs[1]].feasible = True

    # adjacents - critical poin ================================================

    def find_min_n_adjacents(self):
        min_n = len(self.vps_free) // 10  # at least 10% of the Voronoi vertices
        self.min_n_adjacents = max(self.fixed_min_n_adj, min_n)

    def calculate_adj_dist_ub(self) -> float:
        # bounds = self.b_poly.bounds
        # width = bounds[2] - bounds[0]
        # height = bounds[3] - bounds[1]
        # mean_wh = (width + height) / 2.0
        # self.lg.debug(f"Calculated mean width/height: {mean_wh}")
        # self.mean_wh = mean_wh
        # dist_threshold = mean_wh / 10.0  # Adjust the divisor as needed
        #
        dist_threshold = 2 * self.robot_r_px
        self.lg.debug(f"Calculated distance threshold: {dist_threshold:.2f}")
        return dist_threshold

    def find_adjacents_by_distance(self, adj_dist_ub: float = 10.0):
        """
        Find adjacent Voronoi vertices based on a distance threshold.
        This method is used to connect Voronoi vertices that are close enough to each other.
        """
        self.lg.info(f"Finding adjacent Voronoi vertices within distance {adj_dist_ub:.2f}.")
        vertex_adj_idxs = defaultdict(list)
        vertex_adj_dists = defaultdict(list)

        for i, v1 in enumerate(self.v_points):
            if not v1.feasible:
                continue
            for j, v2 in enumerate(self.v_points):
                if i == j or not v2.free:  # feasible
                    continue
                dist = np.linalg.norm(v1.v - v2.v)
                vertex_adj_idxs[i].append(j)
                vertex_adj_dists[i].append(dist)

        # for eacah v, sort the adjacent vertices by distance
        for v_idx, adj_idxs in vertex_adj_idxs.items():
            dists = np.array(vertex_adj_dists[v_idx])
            sorted_indices = np.argsort(dists)
            sorted_adj_idxs = [adj_idxs[i] for i in sorted_indices]
            sorted_dists = dists[sorted_indices]
            n_adj = max(np.sum(sorted_dists < adj_dist_ub), self.min_n_adjacents)
            n_adj = min(n_adj, self.fixed_max_n_adj)

            if n_adj == 0:
                self.lg.warn(f"Vertex {v_idx} has no adjacent vertices within the distance threshold.")
                continue

            self.v_points[v_idx].set_adjacent_vp_idxs(sorted_adj_idxs[:n_adj])
            self.v_points[v_idx].set_adjacents_vp_dists(sorted_dists[:n_adj])

    def find_adjacents(self):
        vor = self.vor
        vertex_to_adjacent = defaultdict(set)

        # Step 1: For each ridge, associate the vertices with their adjacent vertices
        for vertex_indices in vor.ridge_vertices:
            if -1 in vertex_indices:
                continue
            for v_ind in vertex_indices:
                # Associate this vertex with the other vertex in the ridge
                if self.v_points[v_ind].free:
                    other_vertex = [v for v in vertex_indices if v != v_ind][0]
                    vertex_to_adjacent[v_ind].add(other_vertex)

        # Step 2: For each vertex, convert the set of adjacent vertices to a list
        dc_v_inds = []  # disconnected Voronoi vertices
        for v_idx, adj_indices in vertex_to_adjacent.items():
            n_in_poly_o1 = 0
            adj_list = []
            for i in list(adj_indices):
                if self.v_points[i].in_poly_o1:
                    n_in_poly_o1 += 1
                if self.v_points[i].free:
                    adj_list.append(i)
            self.v_points[v_idx].set_adjacent_vp_idxs(adj_list)
            #
            n1 = len(adj_list)
            if self.v_points[v_idx].free:
                if n_in_poly_o1 != n1 and n1 == 1:
                    dc_v_inds.append(v_idx)
        return dc_v_inds

    def find_critical_points(self):
        """ critical points: voronoi vertices with minimum clearance to obstacles."""
        self.lg.info("Finding critical points in Voronoi vertices.")
        critical_idxs = []

        for vp in self.v_points:

            if not vp.feasible:
                continue

            if vp.has_inf_adj:
                continue

            adj_clearances = []
            for adj_idx in vp.adjacent_vp_idxs:
                adj_vp = self.v_points[adj_idx]

                if not adj_vp.free:  # feasible toeval todo
                    continue

                # if vp has this kind of adjacent vertex, it is not set as critical
                if adj_vp.critical or adj_vp.has_inf_adj:
                    adj_clearances = [0]
                    break

                adj_clearances.append(adj_vp.clearance)
                if len(adj_clearances) == 0:
                    raise ValueError("[find_critical_points]: No adjacent Voronoi vertices found!")

            if vp.clearance <= np.min(adj_clearances):

                # check clearance differences
                # if the clearance difference is less than the threshold, skip this vertex
                c_diffs = np.abs(np.array(adj_clearances) - vp.clearance)
                if max(c_diffs) < self.d_clearance_thresh_p * vp.clearance:
                    continue

                # check if it is root
                is_root = False
                for ridge_vp in vp.ridge_vps:
                    if ridge_vp.n_feas_ridges < 2:
                        is_root = True
                if not is_root:
                    critical_idxs.append(vp.idx)

                #
                vp.critical = True
        self.critical_idxs = critical_idxs

    def create_critical_points_list(self):
        """ Create a list of critical points from the Voronoi vertices. """
        for idx in self.critical_idxs:
            vp = self.v_points[idx]
            cp = CriticalPoint(vp)
            self.critical_points.append(cp)

    # ==========================================================================

    def process_polygon(self, poly_t: Polygon, reverse: bool = False):
        poly = self.process_polygon_1(poly_t)
        poly = self.process_polygon_2(poly)
        if reverse:
            poly = poly.reverse()
        return poly
        # method = 2
        # if method == 1:
        # elif method == 2:
        # elif method == 3:
        #     return self.process_polygon_3(poly)
        # else:
        #     raise ValueError(f"Unknown polygon processing method: {method}.")

    def process_polygon_1(self, poly: Polygon):
        """ check distance between points of the polygon.
            add points if the distance between them is greater than a threshold.
        """
        # get points of the polygon and ists
        # min_dist = self.min_poly_points_dist
        robot_r_px = self.robot_r_px
        robot_r_px = 2 * int(robot_r_px)
        min_dist = max(robot_r_px, 1)
        points = np.array(poly.exterior.coords)
        dists = np.linalg.norm(points[:-1] - points[1:], axis=1)
        if np.any(dists > min_dist):
            new_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                dist = np.linalg.norm(p1 - p2)
                new_points.append(Point(points[i]))
                if dist > min_dist:
                    n_points = int(np.ceil(dist / min_dist))
                    for j in range(1, n_points-1):
                        new_point = p1 + (p2 - p1) * (j / n_points)
                        new_points.append(Point(new_point))
            new_points.append(Point(points[-1]))
            # add new points to the polygon
            new_poly = Polygon(new_points)
            return new_poly
        else:
            return poly

    def process_polygon_2(self, poly: Polygon):
        """  if edge is larger than neighboring edges, add points between them """
        points = np.array(poly.exterior.coords)
        points = np.insert(points, 0, points[-1], axis=0)  # add last point to the beginning
        points = np.append(points, [points[1]], axis=0)  # add second point to the end
        dists = np.linalg.norm(points[:-1] - points[1:], axis=1)
        n_dists = len(dists)
        new_points = []
        idx = 0
        for dist in dists[1:n_dists-1]:
            idx += 1
            new_points.append(Point(points[idx]))
            d_prev = dists[idx-1]
            d_next = dists[idx+1]
            k = 1.2
            if (dist > k*d_prev) and (dist > k*d_next):
                d_new = (d_prev + d_next) / 2.0
                n = int(np.ceil(dist / d_new))
                n = min(n, 4)  # limit the number of new points to 10
                for j in range(1, n):
                    new_point = points[idx] + (points[idx+1] - points[idx]) * (j / n)
                    new_points.append(Point(new_point))
        new_points.append(Point(points[n_dists-1]))
        new_poly = Polygon(new_points)
        return new_poly

    def process_polygon_3(self, poly: Polygon):
        """add point between each two points of the polygon
        """
        min_l = np.inf
        points = np.array(poly.exterior.coords[:-1])
        new_points = []
        for i in range(len(points) - 1):
            new_points.append(Point(points[i]))
            p1 = points[i]
            p2 = points[i + 1]
            dist = np.linalg.norm(p1 - p2)
            if dist < min_l:
                min_l = dist
            new_point = (p1 + p2) / 2.0
            new_points.append(Point(new_point))
        new_points.append(Point(points[-1]))
        #
        new_poly = Polygon(new_points)
        self.min_poly_points_dist = min(self.min_poly_points_dist, min_l)
        return new_poly

    def update_min_min_f_edge_l(self):
        min_dist = self.min_poly_points_dist
        for v_idxs in self.vor.ridge_vertices:
            i1 = v_idxs[0]
            i2 = v_idxs[1]
            c1 = self.v_points[i1].free and (not self.v_points[i1].feasible)
            c2 = self.v_points[i2].free and (not self.v_points[i2].feasible)
            if c1 or c2:
                dist = np.linalg.norm(self.v_points[i1].v - self.v_points[i2].v)
                if dist < min_dist:
                    min_dist = dist
        if self.min_poly_points_dist > min_dist * 2.0:
            self.lg.info(f"min_dist: {min_dist:.2f}, min_poly_points_dist: {self.min_poly_points_dist:.2f}")
            self.min_poly_points_dist = self.min_poly_points_dist * 0.5
            # self.min_poly_points_dist = min_dist * 2.0
            return False
        return True

    # ==========================================================================

    def plot_voronoi(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_aspect('equal', adjustable='box')
        x_min = self.b_poly.bounds[0]
        x_max = self.b_poly.bounds[2]
        y_min = self.b_poly.bounds[1]
        y_max = self.b_poly.bounds[3]
        ax.set_xlim(x_min - 10, x_max + 10)
        ax.set_ylim(y_min - 10, y_max + 10)
        ax.set_title('Voronoi Diagram')

        # plot the bounding polygon
        x, y = self.b_poly.exterior.xy
        ax.plot(x, y, color='black', linewidth=2, label='Bounding Polygon')
        # plot the polygons
        for poly in self.obs_polys:
            x, y = poly.exterior.xy
            ax.plot(x, y, color='black', linewidth=1, label='Obstacle Polygon')
            # ax.fill(x, y, alpha=0.5, fc='lightgray', ec='black')

        # plot Voronoi diagram
        voronoi_plot_2d(self.vor, ax=ax, show_vertices=False, line_colors='orange', line_width=1, line_alpha=0.6)

        # plot Voronoi vertices
        for vp in self.v_points:
            if vp.feasible:
                ax.plot(vp.v[0], vp.v[1], 'o', color='red', markersize=2)
                # ax.annotate(f"[{vp.idx}]{int(vp.clearance)}", (vp.v[0], vp.v[1]), fontsize=6, color='black', ha='right', va='bottom')
                # ax.annotate(vp.idx, (vp.v[0], vp.v[1]), fontsize=8, color='black', ha='right', va='bottom')

        # plot basis points
        for vp in self.v_points:
            if vp.basis_idxs is not None:
                ax.plot(vp.basis_points[:, 0], vp.basis_points[:, 1], 'x', color='blue', markersize=2, label='Basis Points')

        # plot critical vertices
        for idx in self.critical_idxs:
            vp = self.v_points[idx]
            ax.plot(vp.v[0], vp.v[1], 'o', color='green', markersize=5, label='Critical Vertex')
            # connect critical vertices to their basis points
            if vp.basis_points is None:
                continue
                # raise ValueError(f"Basis points for critical vertex {idx} are not set.")
            for basis_point in vp.basis_points:
                ax.plot([vp.v[0], basis_point[0]], [vp.v[1], basis_point[1]], color='blue', linewidth=2.5, label='Critical Basis Line')

        # # plot points
        # for i, p in enumerate(self.vor.points):
        #     # ax.plot(p[0], p[1], 'o', color='black', markersize=3, label='Voronoi Point')
        #     ax.annotate(f"{i}", (p[0], p[1]), fontsize=6, color='black', ha='right', va='bottom')

        # # plot adjacency lines
        # for vp in self.v_points:
        #     if not vp.free:
        #         continue
        #     if vp.adjacent_vp_idxs is not None:
        #         for adj_idx in vp.adjacent_vp_idxs:
        #             adj_vp = self.v_points[adj_idx]
        #             ax.plot([vp.v[0], adj_vp.v[0]], [vp.v[1], adj_vp.v[1]], color='green', linewidth=0.5, label='Adjacency Line')

        # # plot Voronoi regions
        # for region in self.vor.regions:
        #     if not -1 in region and len(region) > 0:
        #         polygon = self.vor.vertices[region]
        #         ax.fill(*zip(*polygon), alpha=0.3, fc='blue', ec='black', label='Voronoi Region')

        #
        save_path = self.zmd.result_path + "/voronoi.png"
        fig.savefig(save_path, dpi=300)

    # ==========================================================================
    # self.find_adjacents()
    # trees, troots = self.create_trees()
    # self.connect_trees(troots)

    def connect_trees(self, troots: list[list[int]]) -> None:
        # create distance matrix between troots elements
        n_troots = len(troots)
        len_troots = [len(troot) for troot in troots]
        bigest_tree_idx = np.argmax(len_troots)
        dist_matrix = np.ones((n_troots, n_troots)) * np.inf
        connection_matrix = np.empty((n_troots, n_troots), dtype=object)
        for i, troot_i in enumerate(troots):
            for j, troot_j in enumerate(troots):
                if i == j:
                    continue
                # calculate distance between troot_i and troot_j
                for idx_i in troot_i:
                    for idx_j in troot_j:
                        dist = np.linalg.norm(self.v_points[idx_i].v - self.v_points[idx_j].v)
                        if dist < dist_matrix[i, j]:
                            dist_matrix[i, j] = dist
                            connection_matrix[i, j] = (idx_i, idx_j)
        # connect troots
        connections = []
        comp = [bigest_tree_idx]
        while len(comp) < 2*len(troots):
            min_dist = np.inf
            min_i = -1
            min_j = -1
            for i in comp:
                for j in range(len(troots)):
                    if j in comp:
                        continue
                    if dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        min_i = i
                        min_j = j
            if min_i == -1 or min_j == -1:
                break
            connections.append((min_i, min_j))
            comp.append(min_j)

        # update adjacency lists
        for i, j in connections:
            idx_i, idx_j = connection_matrix[i, j]
            self.v_points[idx_i].adjacent_vp_idxs.append(idx_j)
            self.v_points[idx_j].adjacent_vp_idxs.append(idx_i)

    def create_trees(self):
        visited = set()
        trees = []
        troots = []
        for vp in self.v_points:
            if not vp.free:
                continue
            if vp.idx not in visited:
                tree = []
                troot = []
                self.depth_first_search(vp, tree, visited, troot)
                trees.append(tree)
                troots.append(troot)
        return trees, troots

    def depth_first_search(self, start_vp: VPoint, tree: list, visited: set, troot: list):
        tree.append(start_vp.idx)
        visited.add(start_vp.idx)
        is_last = True
        for adj_idx in start_vp.adjacent_vp_idxs:
            if adj_idx not in visited:
                is_last = False
                self.depth_first_search(self.v_points[adj_idx], tree, visited, troot)
        if is_last:
            troot.append(start_vp.idx)

    # ==========================================================================
