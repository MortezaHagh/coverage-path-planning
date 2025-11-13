""" Coverage Path Planning - Local Energy Minimization Module """
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.base_utils import angle_diff, MyPoint, MyPath
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.voronoi_segmentation.cell import Cell
from covpp.planning.coverage_pp_base import CoveragePathPlannerBase
from covpp.planning.local_energy_minimization.grid import PGrid, GridGenerator, GridNode


class CoveragePathPlanner(CoveragePathPlannerBase):
    """Coverage Path Planning - Local Energy Minimization Module

    Implements a local energy minimization strategy for generating coverage paths
    within a given cell.

    Result:
        Sets path_m and path_px attributes for the cell.
    """

    def __init__(self, cell: Cell, zmd: ZoneMapData):
        super().__init__(cell, zmd)

        self.lg = MyLogger(f"CovPPLocalEnergyMin-Z{zmd.ind}-C{cell.id}")

        # attributes
        self.grid: PGrid = GridGenerator(cell=self.cell, zmd=zmd).grid

        # start coverage path planning for the input cell and map
        self.coverage_planning()

    def coverage_planning(self):

        # create path in grid map
        path_px = self.create_path_in_grid_map()

        # move path from grid map to original map
        # xy = self.cell.global_xy
        # path_px = [(int(p[0] + xy[0]), int(p[1] + xy[1])) for p in path_px]
        path_px = [MyPoint(p[0], p[1]) for p in path_px]

        # set paths
        self.path_px = path_px
        self.path_px = MyPath(path_px)
        self.create_path_m()

    def create_path_in_grid_map(self) -> list[tuple]:
        n_visited = 0
        path_nodes: list[GridNode] = []
        cur_node: GridNode = self.grid.current_node
        cur_heading: float = self.grid.current_heading

        while cur_node.visited is False:
            # mark current node as visited
            cur_node.visited = True
            n_visited += 1
            path_nodes.append(cur_node)

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

            # if no next node found, break
            if next_node is None:
                break
            cur_node = next_node
            cur_heading = next_heading

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
        distance = np.linalg.norm(dxy) / self.grid.grid_l  # normalize by grid size

        # compute heading change
        move_heading = np.arctan2(dxy[1], dxy[0])
        d_heading = angle_diff(heading, move_heading)
        d_heading = abs(d_heading) / np.pi/2  # normalize by pi/2

        # total cost
        cost = distance  # + abs(d_heading)  # + self.cal_track_cost(to_node)
        cost = abs(d_heading)
        return (cost, move_heading)

    def cal_track_cost(self, node: GridNode) -> float:
        n_visited = 0
        for adj in node.adj_nodes:
            if adj.visited:
                n_visited += 1
        score = (node.n_adj - n_visited)/2
        return score
