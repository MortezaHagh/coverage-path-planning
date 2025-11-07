"""Boustrophedon (back-and-forth) coverage path planning implementation."""
import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.base.my_utils import MyPath, MyPoint
from scripts.planning.boustrophedon.cell import Cell
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.sequence.connect_paths import ConnectPaths
from scripts.planning.coverage_pp_base import CoveragePathPlannerBase
from scripts.planning.boustrophedon.cell_cov_planner import CellCoveragePathPlanner
from scripts.planning.boustrophedon.decomposition_bs_mh import CellDecomposition
# from shapely import affinity  # , Point


class CoveragePathPlanner(CoveragePathPlannerBase):
    """Coverage path planner using boustrophedon (back-and-forth) decomposition.

    This planner creates parallel sweeping paths by decomposing the cell into
    trapezoidal regions and generating alternating scan lines.

    Pipeline Steps:
        1. Create map polygons for cell workspace
        2. Decompose cell into boustrophedon sub-cells
        3. Plan coverage path for each sub-cell
        4. Rotate paths back to original orientation
        5. Connect all sub-cell paths into complete coverage

    Result:
        Sets path_m and path_px attributes for the cell.
    """

    def __init__(self, cell: Cell, zmd: ZoneMapData):
        super().__init__(cell, zmd)

        self.lg = MyLogger(f"CoveragePlannerBoust-Z{zmd.ind}-C{cell.id}")

        # map polygons
        self.map_polygons: MapPolygons = None
        self.origin: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)

        # cell decomposition
        self.decomposer: CellDecomposition = CellDecomposition(zmd.resolution, zmd.result_dir)

        # connect paths interface
        self.connections: ConnectPaths = ConnectPaths(zone_map_data=self.zmd, ns=f"C{cell.id}")

        # resulting cells after decomposition
        self.mcells: list[Cell] = []    # boustrophedon cells within the given cell
        self.n_mcells: int = 0          # number of boustrophedon cells
        # self.smcells: list[Cell] = []   # sorted boustrophedon cells

        # start coverage path planning for the input cell and map
        self.coverage_planning()

    def coverage_planning(self) -> None:
        """Execute boustrophedon coverage planning for the cell.

        Orchestrates the complete planning process from decomposition to
        connected path generation.
        """
        if not self.cell.do_planning:
            self.lg.warn(f"cell {self.cell.id} does not require planning.")
            return

        # *******************************
        if not self.create_polygons():
            self.cell.do_planning = False
            return
        self.decompose()
        # self.rotate_back_cells()
        self.all_cells_coverage_pp()
        self.rotate_back_paths()
        self.create_connected_path()
        # *******************************

    def create_polygons(self) -> bool:
        """ Create MapPolygons object for the cell area in the map"""

        # get polygon points of the cell
        points = self.cell.points
        if self.dcfgs.use_insc_ret and self.cell.rect_like:
            points = np.array(self.cell.rectangle_poly.exterior.coords, dtype=np.int32)

        # get ws map
        map_ws = self.zmd.map_ws.copy()

        # - Create polygon coordinates relative to map origin
        poly_coords = [points - self.zmd.rectangle.min_point]

        # - Create mask (same size as map)
        mask = np.zeros_like(map_ws, dtype=np.uint8)

        # - Fill polygon area with 1’s
        cv.fillPoly(mask, poly_coords, 1)

        # - Apply mask
        masked_map = np.where(mask == 1, map_ws, 0)

        # create MapPolygons object
        self.map_polygons = MapPolygons(zmd=self.zmd, map_ws=masked_map, dec_mid_dist=30)
        if not self.map_polygons.success:
            self.lg.error("Failed to create map polygons for the cell area.")
            return False

        # get origin (center of bounding polygon)
        self.origin = np.array(self.map_polygons.b_poly.centroid.coords)[0]

        # rotate all polygons to align with cell free angle
        self.map_polygons.rotate_all_polygons(-self.cell.major_direction, self.origin)
        return True

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
            self.lg.error("No cells to plan coverage path.")
            raise ValueError("No cells to plan coverage path.")

        for cell in self.mcells:

            if not cell.poly.is_valid:
                cell.do_planning = False
                self.lg.warn(f"Cell {cell.id} polygon is not valid.")
                continue
            cell_planner = CellCoveragePathPlanner(cell=cell, zone_ind=self.zmd.ind, resolution=self.resolution)
            if cell_planner.path_m:
                cell.set_path_m(cell_planner.path_m)
                cell.set_path_px(cell_planner.path_px)
            else:
                cell.do_planning = False
                # self.lg.warn(f" -- Bouts Cell {cell.id} has no path after coverage planning.")

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

    def rotate_back_paths(self) -> None:
        """ Rotate back all the paths (cell paths and full path) to original orientation. """

        for cell in self.mcells:
            if cell.path_px:
                # rotate back to original orientation
                rot_path = self.rotate_path(cell.path_px, self.cell.major_direction, np.array([0, 0]))
                # translate back to original position
                rot_path = MyPath([MyPoint(int(p.x + self.origin[0]), int(p.y + self.origin[1])) for p in rot_path.points])
                cell.set_path_px(rot_path)
                cell.set_path_m(self.path_px_to_m(rot_path))

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

    # def rotate_back_cells(self) -> None:
    #     """ Rotate back all the cells to original orientation. """
    #     new_cells = []
    #     for cell in self.mcells:
    #         rot_poly = self.map_polygons.rotate_polygon(cell.poly, self.cell.major_direction, origin_t=np.array([0, 0]))
    #         rot_poly = affinity.translate(rot_poly, xoff=self.origin[0], yoff=self.origin[1])
    #         new_cell = Cell(ind=cell.id, poly=rot_poly, event=None)
    #         new_cells.append(new_cell)
    #     self.mcells = new_cells
