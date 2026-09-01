
import cv2 as cv
import numpy as np
from shapely import Polygon
from matplotlib import pyplot as plt
from networkx import Graph, shortest_path
from scipy.ndimage import binary_opening, binary_closing
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.base.geometry import find_angle_between_points
from covpp.base.plot_styles import set_plot_styles, configure_ax
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.skeleton_segmentation.data_structures import CriticalPoint
from covpp.segmentation.skeleton_segmentation.skelet_branch_graph import SkeletBranchGraph as SBG
from covpp.segmentation.skeleton_segmentation.segmented_solution import CreateSegmentationSolution
from covpp.segmentation.skeleton_segmentation.critical_points_operations import CriticalPointOperations

OCCUPIED = 0
FREE = 255


class EspecialBranchPointsSegmentation:

    def __init__(self, zmd: ZoneMapData, polygons_in: list[Polygon]):
        self.lg = MyLogger("EspecialBranchPointsSegmentation")

        # Data
        self.zmd = zmd
        self.segmented_map: np.ndarray = None
        self.polygons_in: list[Polygon] = polygons_in

        # Parameters
        self.robot_radius_px = zmd.robot_radius_px
        self.min_dist_thresh_branch_px = 4 * self.robot_radius_px

        #
        self.polygons: list[Polygon] = None
        self.map_ws_segmented: np.ndarray = None
        self.adjacency_matrix: np.ndarray = None

    # --------------------------------------------------------------------------
    # Graph-Based Segmentation Based on current cells and segmented_map
    # --------------------------------------------------------------------------

    def run(self) -> bool:
        return self.run_whole()

    def run_whole(self) -> bool:
        self.create_maps()
        graph_data, maps = self.create_graphs()
        cr_pts, base_pts = self.create_critical_points(graph_data, maps)
        if base_pts is None or len(base_pts) == 0:
            return False
        # create segmented solution
        css = CreateSegmentationSolution(map_ws=self.zmd.map_ws.copy(),
                                         map_ws_segmented=self.zmd.map_ws_segmented.copy(),
                                         robot_radius_px=self.robot_radius_px,
                                         origin_coord=self.zmd.global_xy)
        css.run_by_map(base_pts=base_pts)
        self.polygons = css.polygons
        self.map_ws_segmented = css.map_ws_segmented
        self.adjacency_matrix = css.adjacency_matrix

        self.plot_new_critical_points(cr_pts=cr_pts, base_pts=base_pts)
        return True

    def run_cell_based(self):
        self.created_segmented_map_from_cells(self.zmd.map_ws_segmented)
        # graphs_data keys: sk_graph, bpts_graph, branches
        # maps keys: skeleton_map_b, distance_map
        for i, poly in enumerate(self.polygons_in):
            graph_data, maps = self.create_cell_graphs(poly=poly, id=i + 1)
            cr_pts, base_pts = self.create_critical_points(graph_data, maps)

    # ==================================================================================================================
    # Create maps and graphs for the current segmented map
    # ==================================================================================================================

    def create_maps(self):
        # sk_map = self.prepare_map_for_skeleton(self.zmd.map_ws)
        self.segmented_map = self.zmd.map_ws_segmented.copy()

        # # debugging, plot the filtered branch points graph
        # SBG.plot_branch_graph(None, None, seg_map)

    def create_graphs(self):

        maps = SBG.create_skeletonized_maps(self.segmented_map, method='medial_axis')  # maps: skeleton_map_b, distance_map
        sk_graph = SBG.create_skeleton_graph(skeleton_map_b=maps["skeleton_map_b"], distance_map=maps["distance_map"])

        end_pts, branch_pts = SBG.create_especial_points(sk_graph)
        especial_pts_dict = {"end_pts": end_pts, "branch_pts": branch_pts, "skipped_end_pts": []}

        branches, bpts_graph = SBG.create_branch_points_graph(sk_graph=sk_graph,
                                                              especial_pts_dict=especial_pts_dict,
                                                              len_thresh_px=self.min_dist_thresh_branch_px)

        bpts_graph = SBG.filter_bpts_graph_by_distance(bpts_graph, maps["distance_map"], self.robot_radius_px)

        graph_data = {
            "sk_graph": sk_graph,
            "bpts_graph": bpts_graph,
            "branches": branches,
        }

        # # debugging
        # SBG.plot_branch_graph(branches, bpts_graph, sk_graph)
        return graph_data, maps

    # ==================================================================================================================
    # Cell level graph creation
    # ==================================================================================================================

    def created_segmented_map_from_cells(self, map_in: np.ndarray) -> np.ndarray:
        """ Create a segmented map where each cell polygon is filled with its unique id. """

        # for each cell polygon, fill the segmented map with the cell id
        gxy = self.zmd.global_xy
        segmented_map = map_in.copy()
        for i, poly in enumerate(self.polygons_in):
            id = i + 1  # Assign unique ID to each polygon
            pts = np.array([[int(p[0] - gxy[0]), int(p[1] - gxy[1])] for p in poly.exterior.coords], dtype=np.int32)
            cv.fillPoly(segmented_map, pts=[pts], color=id)
            cv.drawContours(segmented_map, [pts], 0, OCCUPIED, thickness=1)
        segmented_map[map_in == OCCUPIED] = OCCUPIED
        self.segmented_map = segmented_map

    def create_cell_graphs(self, poly: Polygon, id: int) -> tuple[dict, dict]:

        # create a binary map for the polygon where the polygon is FREE and outside is OCCUPIED
        cell_map = self.segmented_map.copy()
        cell_map = np.where(self.segmented_map == id, 1, 0).astype(np.uint8) * 255
        # cell_map = self.prepare_map_for_skeleton(cell_map)
        if cell_map.sum() == 0:
            raise ValueError(f"Cell map for cell id {id} is empty. Check the segmented map and cell polygons.")

        # create skeleton and distance maps, and the skeleton graph
        maps = SBG.create_skeletonized_maps(cell_map, method='medial_axis')  # maps: skeleton_map_b, distance_map
        sk_graph = SBG.create_skeleton_graph(maps["skeleton_map_b"], maps["distance_map"])

        # create especial points (end points and branch points) and the branch points graph
        end_pts, branch_pts = SBG.create_especial_points(sk_graph)
        especial_pts_dict = {"end_pts": end_pts, "branch_pts": branch_pts, "skipped_end_pts": []}
        branches, bpts_graph = SBG.create_branch_points_graph(sk_graph=sk_graph,
                                                              especial_pts_dict=especial_pts_dict,
                                                              len_thresh_px=self.min_dist_thresh_branch_px)

        bpts_graph = SBG.filter_bpts_graph_by_distance(bpts_graph, maps["distance_map"], self.robot_radius_px)

        graph_data = {"sk_graph": sk_graph,
                      "bpts_graph": bpts_graph,
                      "branches": branches,
                      }

        # # debugging
        # SBG.plot_branch_graph(branches, bpts_graph, cell_map)
        return graph_data, maps

    # ==================================================================================================================
    # Find critical points and base points for new segmentation
    # ==================================================================================================================

    def create_critical_points(self, graphs_data: dict, maps: dict) -> tuple[list[CriticalPoint], list[list[tuple[int, int]]]]:
        """ Perform graph-based segmentation improvement using the provided graphs data. """

        # graphs_data keys: sk_graph, bpts_graph, branches
        bpts_graph = graphs_data["bpts_graph"]
        bpts_graph = SBG.filter_bpts_graph(bpts_graph=bpts_graph, robot_radius_px=self.robot_radius_px)
        grpups = self.find_groups_in_bpts_graph(bpts_graph)

        # create a new critical points for each group
        cr_pts: list[CriticalPoint] = []
        for i, m_group in enumerate(grpups):

            group = self.remove_invalid_edges_bpts_graph(m_group)
            # group = m_group.copy()

            n_g = len(group)
            new_cp = None
            gc_dist = maps["distance_map"][m_group[0][0], m_group[0][1]]

            if n_g > 3:
                new_cp = self.find_modified_critical_points_g4(graphs_data["sk_graph"], group, gc_dist)
                if new_cp is not None:
                    cr_pts.append(new_cp)
            elif n_g == 3:
                new_cp = self.find_modified_critical_points_g3(graphs_data["sk_graph"], group, gc_dist)
                if new_cp is not None:
                    cr_pts.append(new_cp)

            # # debugging: plot new critical point
            # if new_cp is None:
            #     continue
            # _, ax = plt.subplots(1, 1)
            # ax.imshow(self.zmd.map_ws, cmap='gray', alpha=0.5)
            # ax.scatter(m_group[0][1], m_group[0][0], c='purple', marker='s', s=70, label='GC')  # highlight gc purple
            # ax.scatter([p[1] for p in m_group], [p[0] for p in m_group], c='k', marker='o', s=60, label='Group Members')
            # if len(group) > 0:
            #     ax.scatter([p[1] for p in group], [p[0] for p in group], c='r', marker='x', s=40, label='Group Edges')
            # if new_cp is not None:
            #     ax.scatter(new_cp.coord[1], new_cp.coord[0], c='g', marker='o', s=30, label='New Critical Point')
            # ax.set_aspect('equal')
            # ax.legend()
            # plt.show()

        if len(cr_pts) == 0:
            return None, None

        self.lg.info2(f"Found {len(cr_pts)} new critical points for second level segmentation.")

        # Create a CriticalPointOperations instance to find base points for the new critical points
        cpo = CriticalPointOperations(distance_map=maps["distance_map_norm"],
                                      skeleton_map_b=maps["skeleton_map_b"],
                                      branches=graphs_data["branches"])
        cpo.critical_pts = cr_pts
        base_pts = cpo.find_base_points(critical_pts=cr_pts)

        # # debuggin, plot the filtered branch points graph
        # SBG.plot_branch_graph(graphs_data["branches"], graphs_data["bpts_graph"], self.segmented_map)

        # # debugging
        # _, ax = plt.subplots(1, 1)
        # ax.imshow(self.zmd.map_ws, cmap='gray', alpha=0.5)
        # for node in graphs_data["bpts_graph"].nodes:
        #     neighbors = list(graphs_data["bpts_graph"].neighbors(node))
        #     ax.scatter(node[1], node[0], c='b', marker='o')
        #     ax.text(node[1], node[0], f"{graphs_data['bpts_graph'].degree[node]}", color='b')
        #     for neighbor in neighbors:
        #         ax.plot([node[1], neighbor[1]], [node[0], neighbor[0]], c='g')
        # for cp in cr_pts:
        #     cp_coord = cp.coord
        #     ax.scatter(cp_coord[1], cp_coord[0], c='r', marker='x', s=50)
        # for bps in cpo.base_pts:
        #     for bp in bps:
        #         ax.scatter(bp[1], bp[0], c='b', marker='o', s=25)
        # ax.imshow(self.segmented_map, cmap='gray', alpha=0.5)
        # ax.set_aspect('equal')
        # plt.show()

        return cr_pts, base_pts

    def find_groups_in_bpts_graph(self, bpts_graph: Graph) -> list[list[tuple[int, int]]]:
        groups = []
        for node in bpts_graph.nodes:
            deg = bpts_graph.degree[node]
            if deg < 2:
                continue
            group = [node]
            group.extend(list(bpts_graph.neighbors(node)))
            if len(group) < 3:
                continue
            groups.append(group)
        return groups

    def remove_invalid_edges_bpts_graph(self, group) -> list[tuple[int, int]]:
        """ remove edgew s connected to gc if it crosses the contour polygon"""
        gc = group[0]
        new_group = [gc]
        invalid_group = []
        for g in group[1:]:
            # if line crosses the segmentation map (i.e., crosses any occupied pixel)
            if self.check_if_line_crosses_segmentation_map(g, gc):
                invalid_group.append(g)
                continue
            new_group.append(g)

        return new_group

    def find_modified_critical_points_g3(self, sk_graph: Graph, group: list[tuple[int, int]], gc_dist: int) -> CriticalPoint:
        """ Find modified critical point for group of 3 branch points."""

        gc = group[0]
        g1 = group[1]
        g2 = group[2]

        if not self.check_if_line_crosses_segmentation_map(g1, g2):
            return None

        angle = find_angle_between_points(g1, gc, g2)
        if angle > np.deg2rad(120):
            return None

        d1 = np.linalg.norm(np.array(gc) - np.array(g1))
        d2 = np.linalg.norm(np.array(gc) - np.array(g2))

        if d1 < d2:
            gg = g2
        else:
            gg = g1

        path = shortest_path(sk_graph, source=tuple(gc), target=tuple(gg))
        n_path = len(path)
        # new_ind = n_path // 5
        new_ind = gc_dist
        new_ind = int(new_ind * 0.7)  # move closer to gc
        if new_ind >= n_path:
            new_ind = n_path // 2
        new_coord = path[new_ind]
        return CriticalPoint(coord=np.array(new_coord))

    def find_modified_critical_points_g4(self, sk_graph: Graph, group: list[tuple[int, int]], gc_dist: int) -> CriticalPoint:
        gc = group.pop(0)
        n = len(group)
        hits = [0 for _ in group]
        for i in range(n):
            j = (i + 1) % n
            if self.check_if_line_crosses_segmentation_map(group[i], group[j]):
                hits[i] += 1
                hits[j] += 1

        max_hit_idx = np.argmax(hits)
        if hits[max_hit_idx] == 0:
            return None
        gi = group[max_hit_idx]
        path = shortest_path(sk_graph, source=tuple(gc), target=tuple(gi))
        n_path = len(path)
        # new_ind = n_path // 5
        new_ind = gc_dist
        new_ind = int(new_ind * 0.7)  # move closer to gc
        if new_ind >= n_path:
            new_ind = n_path // 2
        new_coord = path[new_ind]
        return CriticalPoint(coord=np.array(new_coord))

    # ==================================================================================================================
    # map utils - segmentation map
    # ==================================================================================================================

    def updates_segmentation_map_with_cr_pts(self, base_pts: list[list[tuple[int, int]]]):
        """ Update the segmentation map with new critical points and base points. """
        seg_map = self.segmented_map.copy()

        # create a line connecting each two base points consecutively
        for bases in base_pts:
            n_bases = len(bases)
            for i in range(n_bases):
                j = (i + 1) % n_bases
                b1 = bases[i]
                b2 = bases[j]
                cv.line(seg_map, (b1[1], b1[0]), (b2[1], b2[0]), OCCUPIED, thickness=2)

    def check_if_line_crosses_segmentation_map(self, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
        """ Check if line between p1 and p2 crosses any segmentation lines in segmented map."""
        num_points = int(np.linalg.norm(np.array(p2) - np.array(p1))) * 2  # sample points along the line
        dy = abs(p2[0] - p1[0])
        dx = abs(p2[1] - p1[1])
        num_points = max(dx, dy)  # DDA: ensures we step one pixel along the dominant axis
        for i in range(num_points + 1):
            t = i / num_points
            x = int(round(p1[1] + t * (p2[1] - p1[1])))
            y = int(round(p1[0] + t * (p2[0] - p1[0])))
            if self.segmented_map[y, x] == OCCUPIED:
                return True
        return False

    @staticmethod
    def prepare_map_for_skeleton(map_in: np.ndarray) -> np.ndarray:
        """ Create a map for skeletonization where the map_in is FREE and outside is OCCUPIED. """
        structure = np.ones((3, 3), dtype=np.uint8)
        sk_map = binary_closing(map_in, structure=structure, iterations=2).astype(np.uint8) * 255
        sk_map = binary_opening(sk_map, structure=structure, iterations=2).astype(np.uint8) * 255
        return sk_map

    # =================================================================================================================
    # utilities
    # =================================================================================================================

    @staticmethod
    def get_cell_seg_id(cell: CellBase) -> int:
        return cell.id + 1  # start from 1

    def plot_new_critical_points(self, cr_pts: list[CriticalPoint], base_pts: list[list[tuple[int, int]]]):
        """ Plot the new critical points and base points on the workspace map. """
        if not PipeLineConfig.do_plot_segmentation:
            return

        fig_w = 8.27  # A4 width in inches
        fig_h = fig_w * self.zmd.hw[0] / self.zmd.hw[1]
        set_plot_styles(fig_w, fig_h)

        scale_px = min(self.zmd.hw) / 500.0
        marker_size = 20 / scale_px
        fig, ax = plt.subplots(1, 1, constrained_layout=True)

        temp = True
        ax.contour(self.zmd.map_ws, levels=[0.5], colors='black', linewidths=4 / scale_px, zorder=0)
        for cp in cr_pts:
            cp_coord = cp.coord
            ax.scatter(cp_coord[1], cp_coord[0], c='r', marker='x', s=marker_size, label='New Critical Point' if temp else None)
            temp = False
        temp = True
        for bps in base_pts:
            for bp in bps:
                ax.scatter(bp[1], bp[0], c='b', marker='o', s=marker_size // 2, label='Base Point' if temp else None)
                temp = False

        ax.set_aspect('equal')
        ax.legend()
        ax.invert_yaxis()
        configure_ax(ax, '', show_title=False)

        im_format = 'png'
        save_dir = self.zmd.segmentation_method_dir
        fig_path = save_dir + "/ebps" + f".{im_format}"
        fig.savefig(fig_path, format=im_format, dpi=PipeLineConfig.dpi, bbox_inches='tight')

        # plt.show()
        plt.close('all')
