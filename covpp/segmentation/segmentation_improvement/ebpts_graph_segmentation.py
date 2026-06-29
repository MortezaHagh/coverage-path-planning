
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_opening, binary_closing
from networkx import Graph, cycle_basis, shortest_path
from covpp.base.logger import MyLogger
from covpp.district.zone_map_data import ZoneMapData
from covpp.base.geometry import find_angle_between_points
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.skeleton_segmentation.data_structures import CriticalPoint, Branch
from covpp.segmentation.skeleton_segmentation.skelet_branch_graph import SkeletBranchGraph as SBG


OCCUPIED = 0
FREE = 255


class EspecialBranchPointsSegmentation:
    # def __init__(self, segment_solution: SegmentSolution,
    #              bpts_graph: Graph,
    #              map_ws_segmented: np.ndarray,
    #              skelet_distance_map: np.ndarray,
    #              skeleton_graph: Graph,
    #              robot_radius_px: int = 5):
    #     self.lg = MyLogger("SegmentationImprovement")
    #     self.segmented_map = segment_solution.segmented_map
    #     self.bpts_graph = bpts_graph
    #     self.map_ws_segmented = map_ws_segmented
    #     self.sk_graph = skeleton_graph
    #     self.skelet_distance_map = skelet_distance_map
    #     self.robot_radius_px = robot_radius_px

    #     self.new_cps = []

    def __init__(self, zmd: ZoneMapData, cells: list[CellBase]):
        self.lg = MyLogger("EspecialBranchPointsSegmentation")

        # Data
        self.zmd = zmd
        self.segmented_map: np.ndarray = zmd.map_ws_segmented.astype(np.uint8) * 255
        self.cells: list[CellBase] = cells

        # Attributes
        self.bpts_graph: Graph = None

        # Parameters
        self.robot_radius_px = zmd.robot_radius_px
        self.min_dist_thresh_branch_px = 4 * self.robot_radius_px

    # --------------------------------------------------------------------------
    # Graph-Based Segmentation
    # --------------------------------------------------------------------------

    def run(self):
        self.created_segmented_map_from_cells()
        for cell in self.cells:
            self.create_cell_graphs(cell)
            self.graph_based_segmentation()

    def created_segmented_map_from_cells(self):
        """ Create a segmented map where each cell polygon is filled with its unique id. """

        if self.segmented_map is None or self.segmented_map.size == 0:
            raise ValueError("Segmented map is not initialized.")

        # for each cell polygon, fill the segmented map with the cell id
        gxy = self.zmd.global_xy
        self.segmented_map = np.zeros_like(self.segmented_map, dtype=np.uint8)
        for cell in self.cells:
            cell_id = self.get_cell_seg_id(cell)
            pts = np.array([[int(p[0] - gxy[0]), int(p[1] - gxy[1])] for p in cell.points], dtype=np.int32)
            cv.fillPoly(self.segmented_map, pts=[pts], color=cell_id)
            cv.drawContours(self.segmented_map, [pts], 0, OCCUPIED, thickness=1)

    def create_cell_graphs(self, cell: CellBase):
        cell_id = self.get_cell_seg_id(cell)
        cell_map = self.segmented_map.copy()
        cell_map = np.where(self.segmented_map == cell_id, 1, 0).astype(np.uint8) * 255

        sk_map = self.prepare_map_for_skeleton(cell_map)
        if sk_map.sum() == 0:
            raise ValueError(f"Cell map for cell id {cell_id} is empty. Check the segmented map and cell polygons.")

        sk_graph = SBG.create_skeleton_graph(sk_map, method='skeletonize')

        end_pts, branch_pts = SBG.create_especial_points(sk_graph)
        especial_pts_dict = {"end_pts": end_pts, "branch_pts": branch_pts, "skipped_end_pts": []}

        branches, bpts_graph = SBG.create_branch_points_graph(sk_graph=sk_graph,
                                                              especial_pts_dict=especial_pts_dict,
                                                              len_thresh_px=self.min_dist_thresh_branch_px)
        self.bpts_graph = bpts_graph

        SBG.plot_branch_graph(branches, bpts_graph, cell_map)

    # --------------------------------------------------------------------------
    # Utility Methods
    # --------------------------------------------------------------------------

    def prepare_map_for_skeleton(self, map_in: np.ndarray) -> np.ndarray:
        """ Create a map for skeletonization where the map_in is FREE and outside is OCCUPIED. """
        structure = np.ones((3, 3), dtype=np.uint8)
        sk_map = binary_closing(map_in, structure=structure, iterations=2).astype(np.uint8) * 255
        sk_map = binary_opening(sk_map, structure=structure, iterations=2).astype(np.uint8) * 255
        return sk_map

    @staticmethod
    def get_cell_seg_id(cell: CellBase) -> int:
        return cell.id + 1  # start from 1

    # --------------------------------------------------------------------------
    # Segmentation Improvement Methods
    # --------------------------------------------------------------------------

    def graph_based_segmentation(self):
        """ Second level segmentation refinement."""

        self.lg.info2("Running segmentation improvement.\n")
        self.refinement_step = True

        self.filter_loops_bpts_graph()
        self.filter_short_edges_bpts_graph(n_short_neighbors=1)
        self.filter_short_edges_bpts_graph(n_short_neighbors=0)

        # debuggin, plot the filtered branch points graph
        SBG.plot_branch_graph([], self.bpts_graph, self.zmd.map_ws)

        grpups = self.find_groups_in_bpts_graph()

        # create a new critical points for each group
        # new cp is between g1 and each other group member
        new_cps: list[CriticalPoint] = []
        for i, m_group in enumerate(grpups):

            # group = self.remove_invalid_edges_bpts_graph(m_group, segment_solution)
            group = m_group.copy()
            n_g = len(group)
            new_cp = None

            if n_g > 3:
                new_cp = self.find_modified_critical_points_g4(group)
                if new_cp is not None:
                    new_cps.append(new_cp)
            elif n_g == 3:
                new_cp = self.find_modified_critical_points_g3(group)
                if new_cp is not None:
                    new_cps.append(new_cp)

            # # display group and new critical point for debugging # todo
            # print(f"Group {i}")
            # if new_cp is None:
            #     continue
            # _, ax = plt.subplots(1, 1)
            # ax.imshow(self.zmd.map_ws, cmap='gray', alpha=0.5)
            # ax.scatter(m_group[0][1], m_group[0][0], c='purple', marker='s', s=70)  # highlight gc purple
            # ax.scatter([p[1] for p in m_group], [p[0] for p in m_group], c='k', marker='o', s=60)
            # if len(group) > 0:
            #     ax.scatter([p[1] for p in group], [p[0] for p in group], c='r', marker='x', s=40)
            # # plot new critical point
            # if new_cp is not None:
            #     ax.scatter(new_cp.coord[1], new_cp.coord[0], c='g', marker='o', s=30)
            # ax.set_aspect('equal')
            # plt.show()

        self.new_cps.extend(new_cps)
        # n_prev_base_pts = len(self.base_pts)
        # n_new_cps = len(new_cps)
        # self.lg.info2(f"Found {n_new_cps} new critical points for second level segmentation.")

        # for cp in new_cps:
        #     self.get_base_points_by_expand_side_points(cp, n_pts=2)

        # validate
        # n_new_base_pts = len(self.base_pts)
        # self.lg.info2(f"Prev base points: {n_prev_base_pts}")
        # self.lg.info2(f"New base points: {n_new_base_pts - n_prev_base_pts}.")

        # # new critical points
        # # plot for debugging # todo
        # _, ax = plt.subplots(1, 1)
        # # display skeleton segmented map with base points and critical points
        # ax.imshow(self.zmd.map_ws, cmap='gray', alpha=0.5)
        # for node in self.bpts_graph.nodes:
        #     neighbors = list(self.bpts_graph.neighbors(node))
        #     ax.scatter(node[1], node[0], c='b', marker='o')
        #     ax.text(node[1], node[0], f"{self.bpts_graph.degree[node]}", color='b')
        #     for neighbor in neighbors:
        #         ax.plot([node[1], neighbor[1]], [node[0], neighbor[0]], c='g')
        # # display critical points and base points
        # for cp in new_cps:
        #     cp_coord = cp.coord
        #     ax.scatter(cp_coord[1], cp_coord[0], c='r', marker='x', s=50)
        # for bps in self.base_pts[n_prev_base_pts:]:
        #     for bp in bps:
        #         ax.scatter(bp[1], bp[0], c='b', marker='o', s=25)
        # ax.imshow(segment_solution.segmented_map, cmap='gray', alpha=0.5)
        # ax.set_aspect('equal')
        # plt.show()

        # return self.create_segmented_map()

    # # ==================================================================================================================
    # # second level segmentation methods
    # # ==================================================================================================================

    def filter_loops_bpts_graph(self):
        """ Modify branch ends graph to remove loops by merging loop points into a new point.

        Steps:
        - Find loops in the graph.
        - For each loop, create a new point as the average of loop points.
        - Connect the new point to the neighbors of the loop points.
        - Remove loop points from the graph.
        """

        bps_graph = self.bpts_graph
        if bps_graph is None or bps_graph.number_of_nodes() == 0:
            return

        # Recompute cycles on the current graph and guard against overlapping cycles
        loops = list(cycle_basis(bps_graph))
        for loop in loops:
            loop_nodes = [p for p in loop if bps_graph.has_node(p)]

            if len(loop_nodes) not in [3, 4]:
                continue

            # check loop edges sum cost
            edge_cost = 0.0
            for i in range(len(loop_nodes)):
                p1 = loop_nodes[i]
                p2 = loop_nodes[(i + 1) % len(loop_nodes)]
                if bps_graph.has_edge(p1, p2):
                    edge_cost += bps_graph[p1][p2].get('distance', 1.0)
            if edge_cost > 3.0 * self.robot_radius_px:
                continue

            keep_node = loop_nodes[0]
            for p in loop_nodes[1:]:
                if p == keep_node:
                    continue
                # Reconnect all neighbors of p (outside the loop) to keep_node
                neighbors = list(bps_graph.neighbors(p))
                for neighbor in neighbors:
                    if neighbor not in loop and neighbor != keep_node:
                        bps_graph.add_edge(keep_node, neighbor)
                        dist = np.linalg.norm(np.array(keep_node) - np.array(neighbor))
                        bps_graph[keep_node][neighbor]['distance'] = dist
                        # bps_graph[keep_node][neighbor]['type'] = ''
                # Remove the merged node
                if bps_graph.has_node(p):
                    bps_graph.remove_node(p)
            # Remove any edges in the loop except those connected to keep_node
            for p in loop_nodes[1:]:
                if bps_graph.has_edge(keep_node, p):
                    bps_graph.remove_edge(keep_node, p)

        self.bpts_graph = bps_graph

    def filter_short_edges_bpts_graph(self, n_short_neighbors: int = 1):
        """ Remove short edges in the branch points graph by merging nodes. """

        bps_graph = self.bpts_graph
        if bps_graph is None or bps_graph.number_of_nodes() == 0:
            return

        # For each node, collect all neighbors connected by short edges
        dist_thresh = 4.0 * self.robot_radius_px
        nodes = list(bps_graph.nodes)
        for u in nodes:
            if not bps_graph.has_node(u):
                continue
            short_neighbors = []
            for v in list(bps_graph.neighbors(u)):
                if not bps_graph.has_edge(u, v):
                    continue
                dist = bps_graph[u][v].get('distance', 1.0)
                if dist < dist_thresh:
                    short_neighbors.append(v)
            if len(short_neighbors) > n_short_neighbors:
                for v in short_neighbors:
                    if not bps_graph.has_node(v):
                        continue
                    neighbors_v = list(bps_graph.neighbors(v))
                    for neighbor in neighbors_v:
                        if neighbor != u:
                            bps_graph.add_edge(u, neighbor)
                            new_dist = np.linalg.norm(np.array(u) - np.array(neighbor))
                            bps_graph[u][neighbor]['distance'] = new_dist
                    # Remove the merged node
                    if bps_graph.has_node(v):
                        bps_graph.remove_node(v)

        self.bpts_graph = bps_graph

    def find_groups_in_bpts_graph(self) -> list[list[tuple[int, int]]]:

        groups = []
        for node in self.bpts_graph.nodes:
            deg = self.bpts_graph.degree[node]
            if deg < 2:
                continue
            group = [node]
            group.extend(list(self.bpts_graph.neighbors(node)))
            if len(group) < 3:
                continue
            groups.append(group)

        # groups
        # plot for debugging # todo
        _, ax = plt.subplots(1, 1)
        ax.imshow(self.segmented_map, cmap='gray', alpha=0.5)
        for i, group in enumerate(groups):
            ys = [p[0] for p in group]
            xs = [p[1] for p in group]
            ax.scatter(xs, ys, c='g', marker='o', s=25)
            # first point
            ax.scatter(group[0][1], group[0][0], c='r', marker='x', s=50)
            ax.text(group[0][1], group[0][0], f"{i}", color='r')
        ax.set_aspect('equal')
        ax.set_title("Groups in Branch Points Graph")
        plt.show()

        return groups

    # def remove_invalid_edges_bpts_graph(self, group, segment_solution) -> list[tuple[int, int]]:
    #     """ remove edgew s connected to gc if it crosses the contour polygon"""

    #     # find segment id of gc
    #     gc = group[0]
    #     gc_seg_id = segment_solution.segmented_map[gc[0], gc[1]]
    #     gc_seg_idx = segment_solution.seg_ids.index(gc_seg_id)
    #     cc = segment_solution.polygons[gc_seg_idx].reshape(-1, 2)
    #     gc_poly = Polygon(cc)

    #     new_group = [gc]
    #     invalid_group = []
    #     for g in group[1:]:
    #         # if line actually crosses the polygon
    #         line_gi = LineString([(g[1], g[0]), (gc[1], gc[0])])
    #         if line_gi.crosses(gc_poly):
    #             invalid_group.append(g)
    #             continue
    #         new_group.append(g)

    #     # # check each edge
    #     # # create combination of non-repeating pairs of points in group
    #     # invalid_group = []
    #     # new_group = []
    #     # crossed_lines = []
    #     # n = len(group)
    #     # idxs = list(range(1, n))
    #     # pair_idxs = list(combinations(idxs, 2))
    #     # valid = [True for _ in range(n)]
    #     # for i, j in pair_idxs:
    #     #     if valid[i] and valid[j]:
    #     #         g1 = group[i]
    #     #         g2 = group[j]
    #     #         line_gi = LineString([(g1[1], g1[0]), (g2[1], g2[0])])
    #     #         if line_gi.crosses(gc_poly):
    #     #             crossed_lines.append((g1, g2))
    #     #             valid[i] = False
    #     #             valid[j] = False
    #     #             invalid_group.append((g1, g2))
    #     # new_group = [group[i] for i in range(n) if valid[i]]

    #     # # plot the polygon and group for debugging # todo
    #     # _, ax = plt.subplots(1, 1)
    #     # ax.imshow(self.map_ws_segmented, cmap='gray', alpha=0.5)
    #     # ax.scatter([p[1] for p in group], [p[0] for p in group], c='r', marker='x', s=60)
    #     # x_poly = gc_poly.exterior.xy[0]
    #     # y_poly = gc_poly.exterior.xy[1]
    #     # ax.plot(x_poly, y_poly, c='b')
    #     # ax.scatter(gc[1], gc[0], c='purple', marker='s', s=100)  # highlight gc purple
    #     # # plot invalid group
    #     # if len(invalid_group) > 0:
    #     #     ax.scatter([p[1] for p in invalid_group], [p[0] for p in invalid_group], c='y', marker='s', s=25)
    #     # # plot new group
    #     # if len(new_group) > 0:
    #     #     ax.scatter([p[1] for p in new_group], [p[0] for p in new_group], c='g', marker='o', s=25)
    #     # ax.set_aspect('equal')
    #     # plt.show()
    #     return new_group

    # def find_modified_critical_points_g3(self, group: list[tuple[int, int]]) -> CriticalPoint:
    #     """ Find modified critical point for group of 3 branch points."""

    #     gc = group[0]
    #     g1 = group[1]
    #     g2 = group[2]

    #     if not self.check_if_line_crosses_segmentation_map(g1, g2):
    #         return None

    #     angle = find_angle_between_points(g1, gc, g2)
    #     if angle > np.deg2rad(120):
    #         return None

    #     d1 = np.linalg.norm(np.array(gc) - np.array(g1))
    #     d2 = np.linalg.norm(np.array(gc) - np.array(g2))

    #     if d1 < d2:
    #         gg = g2
    #     else:
    #         gg = g1

    #     path = shortest_path(self.sk_graph, source=tuple(gc), target=tuple(gg))
    #     # n_path = len(path)
    #     # new_ind = n_path // 5
    #     new_ind = self.skelet_distance_map[gc[0], gc[1]]
    #     # new_ind = int(new_ind * 0.7)  # move closer to gc
    #     new_coord = path[new_ind]

    #     # # plot for debugging
    #     # _, ax = plt.subplots(1, 1)
    #     # ax.imshow(self.map_ws_segmented, cmap='gray', alpha=0.5)
    #     # # plot gc, g1, g2
    #     # ax.scatter(gc[1], gc[0], c='purple', marker='s', s=100)  # highlight gc purple
    #     # ax.scatter(g1[1], g1[0], c='r', marker='x', s=60)
    #     # ax.scatter(g2[1], g2[0], c='r', marker='x', s=60)
    #     # # plot new_coord
    #     # ax.scatter(new_coord[1], new_coord[0], c='g', marker='o', s=40)
    #     # ax.set_aspect('equal')
    #     # ax.set_title("Modified Critical Point for Group of 3 Branch Points")
    #     # plt.show()

    #     return CriticalPoint(coord=np.array(new_coord))

    # def find_modified_critical_points_g4(self, group: list[tuple[int, int]]) -> CriticalPoint:
    #     gc = group.pop(0)
    #     n = len(group)
    #     hits = [0 for _ in group]
    #     for i in range(n):
    #         j = (i + 1) % n
    #         if self.check_if_line_crosses_segmentation_map(group[i], group[j]):
    #             hits[i] += 1
    #             hits[j] += 1

    #         # # check if line between gi and gj is inside polygon
    #         # line_ij = LineString([(group[i][1], group[i][0]), (group[j][1], group[j][0])])
    #         # if not gc_poly.contains(line_ij):
    #         #     hits[i] += 1
    #         #     hits[j] += 1

    #     max_hit_idx = np.argmax(hits)
    #     if hits[max_hit_idx] == 0:
    #         return None
    #     gi = group[max_hit_idx]
    #     path = shortest_path(self.sk_graph, source=tuple(gc), target=tuple(gi))
    #     n_path = len(path)
    #     # new_ind = n_path // 5
    #     new_ind = self.skelet_distance_map[gc[0], gc[1]]
    #     if new_ind >= n_path:
    #         new_ind = n_path // 2
    #     new_coord = path[new_ind]
    #     return CriticalPoint(coord=np.array(new_coord))

    # def check_if_line_crosses_segmentation_map(self, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
    #     """ Check if line between p1 and p2 crosses any segmentation lines in segmented map."""
    #     # line = LineString([(p1[1], p1[0]), (p2[1], p2[0])]) # todo
    #     num_points = int(np.linalg.norm(np.array(p2) - np.array(p1))) * 2  # sample points along the line
    #     dy = abs(p2[0] - p1[0])
    #     dx = abs(p2[1] - p1[1])
    #     num_points = max(dx, dy)  # DDA: ensures we step one pixel along the dominant axis
    #     for i in range(num_points + 1):
    #         t = i / num_points
    #         x = int(round(p1[1] + t * (p2[1] - p1[1])))
    #         y = int(round(p1[0] + t * (p2[0] - p1[0])))
    #         if self.map_ws_segmented[y, x] == OCCUPIED:
    #             return True
    #     return False

    # @staticmethod
    # def viz_img(img: np.ndarray, name: str):
    #     cv.namedWindow(name, cv.WINDOW_NORMAL)
    #     cv.imshow(name, img)
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
