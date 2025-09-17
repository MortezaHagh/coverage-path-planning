""" boustrophedon coverage planning inside the given cell """
import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.base.my_utils import MyPath  # MyPoint
from scripts.planning.boustrophedon.cell import Cell
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.sequence.connect_paths import ConnectPaths
from scripts.planning.coverage_pp_base import CoveragePathPlannerBase
from scripts.planning.boustrophedon.cell_cov_planner import CellCoveragePathPlanner
from scripts.planning.boustrophedon.decomposition_bs_mh import CellDecomposition


class CoveragePathPlanner(CoveragePathPlannerBase):
    """ Coverage path planning inside the given cell using boustrophedon method. """

    def __init__(self, cell: Cell, zmd: ZoneMapData):
        super().__init__(cell, zmd)

        self.ns = "CoveragePlannerBoust-Z" + str(zmd.ind)
        self.lg = MyLogger(self.ns)

        # map polygons
        self.map_polygons: MapPolygons = None

        # cell decomposition
        self.decomposer: CellDecomposition = CellDecomposition(zmd.resolution, zmd.result_path)

        # connect paths interface
        self.connections: ConnectPaths = ConnectPaths(self.zmd)

        # resulting cells after decomposition
        self.mcells: list[Cell] = []    # boustrophedon cells within the given cell
        self.n_mcells: int = 0          # number of boustrophedon cells
        # self.smcells: list[Cell] = []   # sorted boustrophedon cells

        # start coverage path planning for the input cell and map
        self.coverage_planning()

    def coverage_planning(self) -> None:
        if not self.cell.do_planning:
            self.lg.warn(f"cell {self.cell.id} does not require planning.")
            return

        # ********************
        self.create_polygons()
        self.decompose()
        self.all_cells_coverage_pp()
        self.create_connected_path()
        # ********************

    def create_polygons(self) -> None:
        """ create MapPolygons object for the cell area in the map"""

        # get polygon of the cell
        poly = self.cell.poly
        if self.dcfgs.use_insc_ret and self.cell.rect_like:
            poly = self.cell.rectangle_poly

        # get ws map
        map_ws = self.zmd.map_ws.copy()

        # - Convert polygon to OpenCV format (int32 array)
        s_x = self.zmd.rectangle.x_min
        s_y = self.zmd.rectangle.y_min
        poly_coords = np.array([[(int(x-s_x), int(y-s_y)) for x, y in poly.exterior.coords]], dtype=np.int32)

        # - Create mask (same size as map)
        mask = np.zeros_like(map_ws, dtype=np.uint8)

        # - Fill polygon area with 1’s
        cv.fillPoly(mask, poly_coords, 1)

        # - Apply mask
        masked_map = np.where(mask == 1, map_ws, 0)

        # create MapPolygons object
        self.map_polygons = MapPolygons(zmd=self.zmd, map_ws=masked_map)

    def decompose(self) -> None:
        """ Decompose the cell area into boustrophedon cells. """

        # set bounding polygon
        self.decomposer.set_bounding_polygon(self.map_polygons.b_poly)
        # set obstacle polygons
        self.decomposer.set_obstacle_polygons(self.map_polygons.obs_polys)

        # # # initiate decomposition
        self.decomposer.initiate()

        # resulting boustrophedon cells
        self.mcells = self.decomposer.mcells
        self.n_mcells = self.decomposer.n_mcells

    def all_cells_coverage_pp(self) -> None:
        """ Perform coverage path planning for all the boustrophedon cells. """

        if self.n_mcells == 0:
            self.lg.error(f"No cells to plan coverage path.")
            raise ValueError("No cells to plan coverage path.")

        for i, cell in enumerate(self.mcells):
            cell_planner = CellCoveragePathPlanner(cell=cell, zone_ind=self.zmd.ind, resolution=self.resolution)
            if cell_planner.path_m:
                cell.set_path_m(cell_planner.path_m)
                cell.set_path_px(cell_planner.path_px)
            else:
                cell.do_planning = False
                self.lg.warn(f"Cell {cell.id} has no path after coverage planning.")

    def create_connected_path_simple(self) -> None:
        """ Create a connected path for all the boustrophedon cells. 
            Simple approach: concatenate the paths of all cells in order.
        """
        full_path_px = []
        full_path_m = []
        for cell in self.mcells:
            if cell.path_px:
                full_path_px.extend(cell.path_px.points)
                full_path_m.extend(cell.path_m.points)

        self.path_px = MyPath(full_path_px)
        self.path_m = MyPath(full_path_m)

    def create_connected_path(self) -> None:
        """ Create a connected path for all the boustrophedon cells. 
            Using ConnectPaths class to find optimal connections between cell paths.
        """
        paths_px = []
        for cell in self.mcells:
            if cell.path_px:
                paths_px.append(cell.path_px)
        self.connections.connect_paths(paths_px=paths_px)
        c_paths_px = self.connections.c_paths_px
        # create full path
        self.create_full_path(paths_px, c_paths_px)

    def create_full_path(self, paths_px: list[MyPath], c_paths_px: list[MyPath]) -> None:
        f_path_px = []
        for i in range(len(paths_px)-1):
            f_path_px.extend(paths_px[i].points)
            f_path_px.extend(c_paths_px[i].points)
        f_path_px.extend(paths_px[-1].points)
        #
        self.path_px = MyPath(f_path_px)
        self.path_m = self.path_px_to_m(self.path_px)
