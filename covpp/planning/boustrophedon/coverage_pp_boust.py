"""Boustrophedon (back-and-forth) coverage path planning implementation."""
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.geometry import MyPath, MyPoint
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.connect_paths import ConnectPaths
from covpp.segmentation.map_polygons import MapPolygons
from covpp.planning.coverage_pp_base import CoveragePathPlannerBase
from covpp.planning.boustrophedon.cell import Cell
from covpp.planning.boustrophedon.plotter import BoustPlotter
from covpp.planning.boustrophedon.decomposition_bs_mh import CellDecomposition
from covpp.planning.boustrophedon.cell_cov_planner import CellCoveragePathPlanner
# from shapely import affinity  # , Point


class CoveragePathPlanner(CoveragePathPlannerBase):
    """Coverage path planner using boustrophedon (back-and-forth) decomposition.

    Basic input data for the planner is MapPolygons object created from **cell.map**.
    cell.map boundary is either original cell polygon or inscribed rectangle.

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

        self.lg = MyLogger(f"CovPPBoust-Z{zmd.ind}-C{cell.id}")

        # map polygons
        self.map_polygons: MapPolygons = None
        self.origin: np.ndarray = np.array([0.0, 0.0], dtype=np.float32)

        # cell decomposition
        self.decomposer = CellDecomposition(cell.id, zmd.resolution, zmd.method_results_dir)
        self.boust_plotter = BoustPlotter()

        # connect paths interface
        self.connections = ConnectPaths(zone_map_data=self.zmd, ns=f"C{cell.id}")

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
        if not self.decompose():
            self.cell.do_planning = False
            return
        self.visualize_boust_cells()
        # self.rotate_back_cells()
        self.all_cells_coverage_pp()
        self.rotate_back_paths()
        self.create_connected_path()
        # *******************************

    def create_polygons(self) -> bool:
        """ Create MapPolygons object for the cell area in the map"""

        # create MapPolygons object
        self.map_polygons = MapPolygons(map_ws=self.cell.map, xy=self.cell.global_xy, dec_mid_dist=30)
        if not self.map_polygons.success:
            self.lg.error(f"Failed to create map polygons for the cell {self.cell.id}.")
            return False

        # get origin (center of bounding polygon)
        self.origin = np.array(self.map_polygons.b_poly.centroid.coords)[0]

        # rotate all polygons to align with cell major direction angle
        self.map_polygons.rotate_all_polygons(-self.cell.major_direction, self.origin)
        return True

    def decompose(self) -> bool:
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

        # check if there are any cells
        if self.n_mcells == 0:
            self.lg.error(f"Decomposition for cell {self.cell.id} returned no boust cells to plan coverage path.")
            # raise ValueError("No cells to plan coverage path.")
            return False
        return True

    def all_cells_coverage_pp(self) -> None:
        """ Perform coverage path planning for all the boustrophedon cells. """

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

    def visualize_boust_cells(self):
        self.boust_plotter.visualize(self.decomposer)
