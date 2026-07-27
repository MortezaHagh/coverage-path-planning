
import cv2 as cv
import numpy as np
from networkx import Graph
from matplotlib import pyplot as plt
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
            "distance_map": norm_distance_map
        }
        return maps

    @staticmethod
    def create_skeleton_graph(skeleton_map_b: np.ndarray) -> Graph:
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
                bpts_graph[start][end]['distance'] = np.linalg.norm(np.array(start) - np.array(end))

                path = np.array(path)
                if len(path) < len_thresh_px:
                    continue

                end_type = "branching" if end in branch_pts else "endpoint"
                branches.append(Branch(id=i_branch,
                                       start_point=start,
                                       end_point=end,
                                       points=path,
                                       length=len(path),
                                       end_type=end_type))
                i_branch += 1

        return branches, bpts_graph

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
