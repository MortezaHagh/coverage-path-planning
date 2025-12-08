""" Coverage Path Planning - Local Energy Minimization Module """
import numpy as np
from covpp.segmentation.cell_planning import PCell
from covpp.district.zone_map_data import ZoneMapData
from covpp.base.geometry import angle_diff, MyPoint, MyPath
from covpp.coverage_planning.coverage_pp_base import CoveragePathPlannerBase
from covpp.map_topology.cell_grid import PGrid, GridGenerator, GridNode


class CoveragePathPlanner(CoveragePathPlannerBase):
    """Coverage Path Planning - Local Energy Minimization Module

    Steps:
        1. Create grid representation of the cell.
        2. Initialize starting position and heading.
        3. Iteratively select next grid node to visit based on cost function
           considering heading change and tracking cost.
        4. Backtrace if no feasible next node is found.
        5. Continue until all free grid nodes are visited.

    Result:
        Sets path_m and path_px attributes for the cell.
    """

    def __init__(self, cell: PCell, zmd: ZoneMapData):
        super().__init__(cell, zmd)
        self.lg.update_name(f"CovPPLocalEnergyMin-Z{zmd.ind}-C{cell.id}")

        # attributes
        self.grid: PGrid = GridGenerator(cell=self.cell, zmd=zmd).grid
        self.major_dir: float = (cell.major_direction + np.pi/2) % np.pi

    def coverage_planning(self):
        """Execute coverage path planning using local energy minimization.

        Result: ``self.path_px``/``self.path_m`` trace the grid-based coverage
        order derived from the local energy heuristic.
        """

        # create path in grid map
        path_px = self.create_path_in_grid_map()

        # move path from grid map to original map
        path_px = [MyPoint(p[0], p[1]) for p in path_px]

        # set paths
        self.path_px = path_px
        self.path_px = MyPath(path_px)

    def create_path_in_grid_map(self) -> list[tuple]:
        """ Create coverage path in the grid map using local energy minimization. """
        n_visited = 0
        path_nodes: list[GridNode] = []
        cur_node: GridNode = self.grid.current_node
        cur_heading: float = self.grid.current_heading

        # mark current node as visited
        n_visited += 1
        cur_node.visited = True
        path_nodes.append(cur_node)

        while n_visited < self.grid.n_free:

            # find next node with minimum cost
            min_cost = float('inf')
            next_node: GridNode = None
            next_heading: float = None
            for adj in cur_node.adj_nodes:
                if adj.free and not adj.visited:
                    cost, move_heading = self.cal_cost(cur_node, adj, cur_heading)
                    if cost < min_cost:
                        min_cost = cost
                        next_node = adj
                        next_heading = move_heading

            # if no next node found
            if next_node is None:
                next_node, next_heading = self.backtrace_find_feasible(path_nodes)
                if next_node is None:
                    raise RuntimeError("No feasible next node found during coverage path planning.")

            # move to next node
            cur_node = next_node
            cur_heading = next_heading

            # mark current node as visited
            n_visited += 1
            cur_node.visited = True
            path_nodes.append(cur_node)

        # convert path nodes to path points
        path_px: list[tuple] = []
        for node in path_nodes:
            path_px.append((node.x, node.y))

        # # plot for debugging (map and path)
        # import cv2 as cv
        # debug_map = cv.cvtColor(self.grid.map, cv.COLOR_GRAY2BGR)
        # for node in path_nodes:
        #     cv.circle(debug_map, (int(node.x), int(node.y)), 2, (0, 0, 255), -1)
        # cv.circle(debug_map, (int(path_nodes[0].x), int(path_nodes[0].y)), 1, (0, 0, 255), -1)
        # self.lg.debug(f"Coverage path planning completed. Visited {n_visited} nodes out of {self.grid.n_free} free nodes.")
        # cv.namedWindow("Debug Map", cv.WINDOW_NORMAL)
        # cv.imshow("Debug Map", debug_map)
        # cv.waitKey(0)

        return path_px

    def cal_cost(self, from_node: GridNode, to_node: GridNode, heading: float) -> tuple[float, float]:
        """ Calculate cost of moving from from_node to to_node given current heading. """

        # cost = distance + heading change + tracking cost

        # compute distance
        dxy = np.array([to_node.x - from_node.x, to_node.y - from_node.y], dtype=np.float32)
        # distance = np.linalg.norm(dxy) / self.grid.grid_l  # normalize by grid size

        # compute heading change
        move_heading = np.arctan2(dxy[1], dxy[0])
        d_heading = angle_diff(heading, move_heading)
        d_heading = abs(d_heading) / np.pi/2  # normalize by pi/2

        # total cost
        # cost = distance + abs(d_heading) + self.cal_track_cost(to_node)
        # cost = abs(d_heading) + self.cal_track_cost(to_node)
        cost = abs(d_heading)
        cost += to_node.i_row * 10
        return (cost, move_heading)

    def cal_track_cost(self, node: GridNode) -> float:
        """ Calculate tracking cost for a grid node based on visited adjacent nodes. """
        n_visited = 0
        for adj in node.adj_nodes:
            if adj.visited:
                n_visited += 1
            elif not adj.free:
                n_visited += 1
        score = (node.n_adj - n_visited)/2
        return score

    def backtrace_find_feasible(self, path_nodes: list[GridNode]):
        """ Backtrace to find the first node in the current path with free adjacent. """

        # find the last node in path with free unvisited adjacent
        for node in path_nodes[::-1]:
            # find one free unvisited adjacent
            for adj in node.adj_nodes:
                if adj.free and not adj.visited:
                    d_xy = np.array([adj.x - node.x, adj.y - node.y], dtype=np.float32)
                    move_heading = np.arctan2(d_xy[1], d_xy[0])
                    return (adj, move_heading)
        return (None, None)
