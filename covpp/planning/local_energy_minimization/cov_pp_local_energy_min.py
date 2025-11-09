""" Coverage Path Planning - Local Energy Minimization Module """
import cv2 as cv
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.my_utils import MyPoint, MyPath
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.voronoi_segmentation.cell import Cell
from covpp.planning.coverage_pp_base import CoveragePathPlannerBase


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

        # start coverage path planning for the input cell and map
        self.coverage_planning()

    def coverage_planning(self):
        pass
