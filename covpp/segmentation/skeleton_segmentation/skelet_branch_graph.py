
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from networkx import Graph, cycle_basis
from skimage.morphology import medial_axis, skeletonize
from covpp.base.logger import MyLogger
from covpp.segmentation.skeleton_segmentation.data_structures import Branch

OCCUPIED = 0
FREE = 255


class SkeletBranchGraph:
    """ Class for creating a graph representation of the skeleton of map. """

    lg = MyLogger("SkeletBranchGraph")

    def __init__(self):
        pass

    @staticmethod
    def create_skeletonized_maps(map_ws: np.ndarray, method: str = 'medial_axis') -> dict:
        """ Create skeleton and distance maps from the workspace map. """

        if method == 'medial_axis':
            skeleton_map_b, distance_map = medial_axis(map_ws, return_distance=True)
        elif method == 'skeletonize':
            skeleton_map_b = skeletonize(map_ws, method='lee')
            distance_map = cv.distanceTransform(map_ws, cv.DIST_L2, 5)
        else:
            raise ValueError(f"Unknown skeletonization method: {method}")
        norm_distance_map = (distance_map / distance_map.max() * FREE).astype(np.uint8)

        # # debugging
        # SkeletBranchGraph.viz_img(skeleton_map_b.astype(np.uint8) * 255, "Skeletonized Map")

        maps = {
            "skeleton_map_b": skeleton_map_b,
            "distance_map": distance_map,
            "distance_map_norm": norm_distance_map
        }
        return maps

    @staticmethod
    def create_skeleton_graph(skeleton_map_b: np.ndarray, distance_map: np.ndarray) -> Graph:
        """ Create a graph representation of the skeleton of the cell map. """

        # create graph from skeleton
        sk_graph = Graph()
        ys, xs = np.where(skeleton_map_b)
        nodes = list(zip(ys.tolist(), xs.tolist()))
        node_set = set(nodes)
        sk_graph.add_nodes_from(nodes)

        # Only scan forward 8-connected neighbors so each undirected edge is added once.
        dxy_forward = [(0, 1), (1, -1), (1, 0), (1, 1)]
        for y, x in nodes:
            sk_graph.nodes[(y, x)]['distance'] = distance_map[y, x]
            for dy, dx in dxy_forward:
                neighbor = (y + dy, x + dx)
                if neighbor in node_set:
                    sk_graph.add_edge((y, x), neighbor)
        return sk_graph

    @classmethod
    def create_especial_points(cls, sk_graph: Graph) -> tuple[list[tuple], list[tuple]]:
        """ Create a list of special points (endpoints and branch points) from the skeleton graph. """

        end_pts = [node for node in sk_graph.nodes if sk_graph.degree[node] == 1]
        branch_pts = [node for node in sk_graph.nodes if sk_graph.degree[node] > 2]
        return end_pts, branch_pts

    @classmethod
    def create_branch_points_graph(cls, sk_graph: Graph, especial_pts_dict: dict, len_thresh_px: int) -> tuple[list[Branch], Graph]:
        """ Create a graph from skeleton graph.

        Each edge is between two special nodes, where special nodes are endpoints
        (degree 1) or branch points (degree > 2). Edges are created only for
        nodes that are directly connected in the skeleton graph, without passing
        through other special nodes.
        """

        branches: list[Branch] = []     # list of branches
        bpts_graph = Graph()            # branch points graph
        end_pts = especial_pts_dict["end_pts"]
        skipped_end_pts = especial_pts_dict.get("skipped_end_pts", [])
        branch_pts = especial_pts_dict["branch_pts"]
        special_pts = end_pts + branch_pts

        # validate
        if len(end_pts) + len(branch_pts) != len(set(special_pts)):
            raise ValueError("Duplicate special points found in skeleton graph.")

        def next_nodes(curr, prev):
            """ Get the next nodes in the skeleton graph that is not the previous node. """
            neighbors = [node for node in sk_graph.neighbors(curr) if node != prev]
            if not neighbors:
                raise ValueError(f"No next node found for current node {curr} with previous node {prev}.")
            return neighbors

        # for traversing the skeleton graph, start from endpoints, then branch points
        i_branch = 0
        visited = set()
        for start in special_pts:
            for nxt in next_nodes(start, None):

                # if bpts_graph.has_node(start):
                #     if bpts_graph.degree[start] == sk_graph.degree[start]:
                #         continue

                if (nxt, start) in visited or (start, nxt) in visited:
                    continue
                visited.add((start, nxt))
                visited.add((nxt, start))

                skip = False
                curr = nxt
                prev = start
                path = [start, curr]

                while nxt not in special_pts:
                    if nxt in skipped_end_pts or nxt == start:
                        skip = True
                        break
                    nxts = next_nodes(curr, prev)
                    if len(nxts) != 1:
                        raise ValueError(f"Multiple next nodes found for current node {curr} with previous node {prev}.")
                    nxt = nxts[0]
                    path.append(nxt)
                    prev, curr = curr, nxt

                if skip:
                    continue

                end = nxt
                if end == start:
                    raise ValueError(f"Loop detected in skeleton graph starting from {start}.")

                visited.add((prev, end))
                visited.add((end, prev))

                bpts_graph.add_edge(start, end)
                bpts_graph[start][end]['dist_eps'] = np.linalg.norm(np.array(start) - np.array(end))

                if len(path) < len_thresh_px:
                    continue
                path_arr = np.array(path)

                dists = [sk_graph.nodes[node]['distance'] for node in path]
                max_dist = max(dists)

                end_type = "branching" if end in branch_pts else "endpoint"
                branches.append(Branch(id=i_branch,
                                       start_point=start,
                                       end_point=end,
                                       points=path_arr,
                                       length=len(path),
                                       end_type=end_type,
                                       dists=np.array(dists),
                                       max_dist=max_dist))
                i_branch += 1

        return branches, bpts_graph

    # ==================================================================================================================
    # filtering bpts graph
    # ==================================================================================================================

    @staticmethod
    def filter_bpts_graph_by_distance(bpts_graph: Graph, distance_map: np.ndarray, robot_radius_px: float) -> tuple[list[tuple], list[tuple]]:
        """ Filter the branch points graph to remove nodes that are too close to obstacles based on the distance map. """

        if bpts_graph is None or bpts_graph.number_of_nodes() == 0:
            return bpts_graph

        # find max 'distance' from the graph nodes
        max_distance = 0.0
        for node in bpts_graph.nodes:
            (y, x) = node
            dist_to_obstacle = distance_map[y, x]
            if dist_to_obstacle > max_distance:
                max_distance = dist_to_obstacle

        # todo
        # threshold_distance = max_distance // 4  # param
        threshold_distance = max(1, 1 * robot_radius_px)  # param 4 *

        # Create a list of nodes to remove
        nodes_to_remove = []
        for node in bpts_graph.nodes:
            y, x = node
            dist_to_obstacle = distance_map[y, x]
            if dist_to_obstacle < threshold_distance:
                nodes_to_remove.append(node)

        # Remove the identified nodes from the graph
        bpts_graph.remove_nodes_from(nodes_to_remove)

        return bpts_graph

    @staticmethod
    def filter_bpts_graph(bpts_graph, robot_radius_px: int) -> Graph:
        dist_thresh = 3.0 * robot_radius_px  # param
        bpts_graph = SkeletBranchGraph.filter_loops_bpts_graph(bpts_graph, dist_thresh=dist_thresh)
        dist_thresh = 4.0 * robot_radius_px  # param
        bpts_graph = SkeletBranchGraph.filter_short_edges_bpts_graph(bpts_graph, dist_thresh=dist_thresh, n_short_neighbors=1)
        bpts_graph = SkeletBranchGraph.filter_short_edges_bpts_graph(bpts_graph, dist_thresh=dist_thresh, n_short_neighbors=0)
        return bpts_graph

    @staticmethod
    def filter_loops_bpts_graph(bpts_graph: Graph, dist_thresh: int) -> Graph:
        """ Modify branch ends graph to remove loops by merging loop points into a new point.

        Steps:
        - Find loops in the graph.
        - For each loop, create a new point as the average of loop points.
        - Connect the new point to the neighbors of the loop points.
        - Remove loop points from the graph.
        """

        if bpts_graph is None or bpts_graph.number_of_nodes() == 0:
            return bpts_graph

        # Recompute cycles on the current graph and guard against overlapping cycles
        loops = list(cycle_basis(bpts_graph))
        for loop in loops:
            loop_nodes = [p for p in loop if bpts_graph.has_node(p)]

            if len(loop_nodes) not in [3, 4]:
                continue

            # check loop edges sum cost
            edge_cost = 0.0
            for i in range(len(loop_nodes)):
                p1 = loop_nodes[i]
                p2 = loop_nodes[(i + 1) % len(loop_nodes)]
                if bpts_graph.has_edge(p1, p2):
                    edge_cost += bpts_graph[p1][p2].get('dist_eps', 1.0)
            if edge_cost > dist_thresh:
                continue

            keep_node = loop_nodes[0]
            for p in loop_nodes[1:]:
                if p == keep_node:
                    continue
                # Reconnect all neighbors of p (outside the loop) to keep_node
                neighbors = list(bpts_graph.neighbors(p))
                for neighbor in neighbors:
                    if neighbor not in loop and neighbor != keep_node:
                        bpts_graph.add_edge(keep_node, neighbor)
                        dist = np.linalg.norm(np.array(keep_node) - np.array(neighbor))
                        bpts_graph[keep_node][neighbor]['dist_eps'] = dist
                # Remove the merged node
                if bpts_graph.has_node(p):
                    bpts_graph.remove_node(p)
            # Remove any edges in the loop except those connected to keep_node
            for p in loop_nodes[1:]:
                if bpts_graph.has_edge(keep_node, p):
                    bpts_graph.remove_edge(keep_node, p)
        return bpts_graph

    @staticmethod
    def filter_short_edges_bpts_graph(bpts_graph: Graph, dist_thresh: int, n_short_neighbors: int = 1) -> Graph:
        """ Remove short edges in the branch points graph by merging nodes. """

        if bpts_graph is None or bpts_graph.number_of_nodes() == 0:
            return bpts_graph

        # For each node, collect all neighbors connected by short edges
        nodes = list(bpts_graph.nodes)
        for u in nodes:
            if not bpts_graph.has_node(u):
                continue
            short_neighbors = []
            for v in list(bpts_graph.neighbors(u)):
                if not bpts_graph.has_edge(u, v):
                    continue
                dist = bpts_graph[u][v].get('dist_eps', 1.0)
                if dist < dist_thresh:
                    short_neighbors.append(v)
            if len(short_neighbors) > n_short_neighbors:
                for v in short_neighbors:
                    if not bpts_graph.has_node(v):
                        continue
                    neighbors_v = list(bpts_graph.neighbors(v))
                    for neighbor in neighbors_v:
                        if neighbor != u:
                            bpts_graph.add_edge(u, neighbor)
                            new_dist = np.linalg.norm(np.array(u) - np.array(neighbor))
                            bpts_graph[u][neighbor]['dist_eps'] = new_dist
                    # Remove the merged node
                    if bpts_graph.has_node(v):
                        bpts_graph.remove_node(v)
        return bpts_graph

    # ==================================================================================================================
    # Inline branches detection
    # ==================================================================================================================

    @classmethod
    def find_straight_bpts_graph(cls, map, bpts_graph: Graph):
        """ Two nodes are connected if the direct line between them is free of obstacles. """

        angle_thresh = np.radians(3)
        n = bpts_graph.number_of_nodes()
        connections = []

        for node1 in bpts_graph.nodes:
            start = node1

            for node2 in bpts_graph.neighbors(node1):
                if node2 == node1:
                    continue
                if cls.check_if_line_crosses__map(map, start, node2):
                    continue

                conn = [node1, node2]
                v1 = np.array(node2) - np.array(node1)
                cur = node2
                found = True
                while found:
                    found = False
                    for next in bpts_graph.neighbors(cur):
                        if next not in conn:

                            # check if line crosses map
                            if cls.check_if_line_crosses__map(map, cur, next):
                                continue

                            # check angle
                            v_next = np.array(next) - np.array(start)
                            angle = np.arccos(np.clip(np.dot(v1, v_next) / (np.linalg.norm(v1) * np.linalg.norm(v_next)), -1.0, 1.0))
                            if angle is np.nan:
                                angle = 0.0
                            if abs(angle) > angle_thresh:
                                continue

                            # add next to connection
                            conn.append(next)
                            cur = next
                            found = True
                            break
                if len(conn) > 2:
                    connections.append(conn)

        # sort connections by length
        lc = [len(conn) for conn in connections]
        lc = np.array(lc)
        sorted_indices = np.argsort(lc)[::-1]
        connections = [connections[i] for i in sorted_indices]

        # final_connections
        visited = set()
        final_connections = []
        for conn in connections:
            pairs = [(conn[i], conn[i + 1]) for i in range(len(conn) - 1)]
            # check if any pair has been visited
            if any((min(p), max(p)) in visited for p in pairs):
                continue
            # mark pairs as visited
            for p in pairs:
                visited.add((min(p), max(p)))
            final_connections.append(conn)

        # debugging
        _, ax = plt.subplots(1, 1)
        ax.imshow(map, cmap='gray', alpha=0.5)
        # plot bpts_graph edges
        for i, edge in enumerate(bpts_graph.edges):
            label = "bpts_graph" if i == 0 else None
            start, end = edge
            ax.plot([start[1], end[1]], [start[0], end[0]], c='b',
                    linewidth=1, linestyle='-', marker='o', markersize=5, label=label)
        # plot final connections
        for i, conn in enumerate(final_connections):
            xs = [p[1] for p in conn]
            ys = [p[0] for p in conn]
            ax.plot(xs, ys, '--', linewidth=3)
        ax.set_title("Inline Branch Connections")
        ax.legend()
        plt.show()

    @staticmethod
    def check_if_line_crosses__map(map, p1: tuple[int, int], p2: tuple[int, int]) -> bool:
        """ Check if line between p1 and p2 crosses any segmentation lines in segmented map."""
        num_points = int(np.linalg.norm(np.array(p2) - np.array(p1))) * 2  # sample points along the line
        dy = abs(p2[0] - p1[0])
        dx = abs(p2[1] - p1[1])
        num_points = max(dx, dy)  # DDA: ensures we step one pixel along the dominant axis
        for i in range(num_points + 1):
            t = i / num_points
            x = int(round(p1[1] + t * (p2[1] - p1[1])))
            y = int(round(p1[0] + t * (p2[0] - p1[0])))
            if map[y, x] == OCCUPIED:
                return True
        return False

    # ==================================================================================================================
    # Visualization methods
    # ==================================================================================================================

    @staticmethod
    def plot_branch_graph(branches: list[Branch] = None, bpts_graph: Graph = None, map_ws: np.ndarray = None) -> None:
        _, ax = plt.subplots(1, 1)
        # map
        if map_ws is not None:
            ax.imshow(map_ws, cmap='gray', alpha=0.5)
        # Branches
        if branches is not None:
            colors = plt.cm.get_cmap('tab10', len(branches))
            for i, branch in enumerate(branches):
                label = "branch" if i == 0 else None
                points = branch.points
                xs = [p[1] for p in points]
                ys = [p[0] for p in points]
                ax.plot(xs, ys, c=colors(i), linewidth=2, label=label)
        # bts_graph edges
        if bpts_graph is not None:
            for i, edge in enumerate(bpts_graph.edges):
                label = "bpts_graph" if i == 0 else None
                start, end = edge
                ax.plot([start[1], end[1]], [start[0], end[0]], c='b',
                        linewidth=1, linestyle='--', marker='o', markersize=5, label=label)

        ax.set_aspect('equal')
        ax.set_title("Skeleton and Branch Points Graph")
        ax.legend()
        plt.show()

    @staticmethod
    def viz_img(img: np.ndarray, name: str):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
