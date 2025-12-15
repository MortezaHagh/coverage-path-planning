""" Cell and CellGroup classes for boustrophedon decomposition."""
from shapely import Polygon
from shapely.ops import unary_union
from covpp.base.geometry import evaluate_polygon
from covpp.segmentation.cell_base import CellBase
from covpp.coverage_planning.boustrophedon.events import Event


class Cell (CellBase):
    """ Cell for boustrophedon decomposition. """

    def __init__(self, ind: int, poly: Polygon, event: Event, resolution: float = 0.05):
        # evaluate validity of polygon
        is_valid = evaluate_polygon(poly)[0]
        if not is_valid:
            print(f"[BoustCell][WARN] Cell {ind}: Polygon is invalid. Trying to fix it.")
            poly = poly.buffer(0)
            is_valid = evaluate_polygon(poly)[0]
            if not is_valid:
                raise ValueError(f"[BoustCell][ERROR] Cell {ind}: Polygon is still invalid after fixing.")

        super().__init__(ind, poly, resolution, simplify=False)

        #
        self.group_id: int = None
        self.event: Event = event

        # corners (used in post processing (shift detection))
        self.ld_point: tuple[float, float] = None  # lower left corner xy
        self.lu_point: tuple[float, float] = None  # lower right corner xy
        self.rd_point: tuple[float, float] = None  # upper left corner xy
        self.ru_point: tuple[float, float] = None  # upper right corner xy

        # adjacent and parent cell indices
        self.up_adj_ci = None       # upper adjacent cell index
        self.down_adj_ci = None     # lower adjacent cell index
        self.parent_up_ci = None    # upper parent cell index
        self.parent_down_ci = None  # lower parent cell index
        #
        self.n_adj = 0
        self.n_parent = 0
        #
        self.mid_adjlist = []       # list of mid adjacent cell indices
        self.mid_parentlist = []    # list of mid parent cell indices

        #
        self.grouped = False            # whether the cell is assigned to a group
        self.found_connection = True    # whether a connection to adjacent cell is found (either parent or adjacent)

        # shifts for cov planning
        self.shift_r: bool = False
        self.shift_l: bool = False

        # ----------------------------------------------------------------------
        # find corners
        self.find_corner()

    def set_group_id(self, group_id: int):
        """ Set the group ID for the cell. """
        if group_id is None:
            raise ValueError("Group ID cannot be None")
        self.group_id = group_id

    def find_corner(self):
        """ Find the corner points of the cell polygon. """
        # left points
        lps = [xy for xy in self.points if xy[0] == int(self.x_min_p)]
        lps = sorted(lps, key=lambda xy: xy[1])
        self.ld_point = lps[0]
        self.lu_point = lps[-1]
        # right points
        rps = [xy for xy in self.points if xy[0] == int(self.x_max_p)]
        rps = sorted(rps, key=lambda xy: xy[1])
        self.rd_point = rps[0]
        self.ru_point = rps[-1]

    def set_up_adj_ci(self, ci: int):
        """ Set the upper adjacent cell index. """
        if ci is not None:
            self.up_adj_ci = ci
            self.add_mid_adjlist(ci)

    def set_down_adj_ci(self, ci: int):
        """ Set the lower adjacent cell index. """
        if ci is not None:
            self.down_adj_ci = ci
            self.add_mid_adjlist(ci)

    def add_mid_adjlist(self, ci: int):
        """ Add a cell index to the mid adjacent list. """
        if ci is not None:
            self.mid_adjlist.append(ci)
            self.n_adj += 1
            self.add_adjacent_id(ci)

    def set_parent_up_ci(self, ci: int):
        """ Set the upper parent cell index. """
        if ci is not None:
            if self.parent_up_ci is None:
                self.parent_up_ci = ci
            self.add_mid_parentlist(ci)

    def set_parent_down_ci(self, ci: int):
        """ Set the lower parent cell index. """
        if ci is not None:
            if self.parent_down_ci is None:
                self.parent_down_ci = ci
            self.add_mid_parentlist(ci)

    def add_mid_parentlist(self, ci: int):
        """ Add a cell index to the mid parent list. """
        if ci is not None:
            self.mid_parentlist.append(ci)
            self.n_parent += 1
            self.add_adjacent_id(ci)

    def set_shift_re(self, shift: bool):
        """ Set the shift right flag. """
        self.shift_r = shift

    def set_shift_le(self, shift: bool):
        """ Set the shift left flag. """
        self.shift_l = shift

# ------------------------------------------------------------------------------
# CellGroup
# ------------------------------------------------------------------------------


class CellGroup:
    """ Group of adjacent cells for boustrophedon decomposition. """
    _counter = 0

    def __init__(self, first_cell: Cell, ncg: int = None):
        if ncg is None:
            self.id = CellGroup._counter
            self.opt_id = CellGroup._counter
        else:
            self.id = ncg
            self.opt_id = ncg
        CellGroup._counter += 1
        self.cells: list[Cell] = [first_cell]   # list of cells in the group
        self.first_cell: Cell = first_cell      # first cell in the group
        self.last_cell: Cell = None             # last cell in the group
        self.merged_cell: Cell = None           # merged cell of the group

    def add_cell(self, cell: Cell):
        """ Add a cell to the group. """
        if cell:
            self.cells.append(cell)

    def set_first_cell(self, cell: 'Cell'):
        """ Set the first cell in the group. """
        if cell:
            self.first_cell = cell

    def set_last_cell(self, cell: 'Cell'):
        """ Set the last cell in the group. """
        if cell:
            self.last_cell = cell

    def update_group(self):
        """ Update the group by setting the last cell and merging cells. """
        if self.cells:
            self.set_last_cell(self.cells[-1])
            self.merge_cells()

    def merge_cells(self):
        """ Merge all cells in the group into a single cell. """
        # Merge all polygons
        if self.cells:
            merged_polygon = unary_union([cell.poly for cell in self.cells])

            # check validity
            is_valid, status = evaluate_polygon(merged_polygon)
            if not is_valid:
                print(f"[CellGroup][WARN] CellGroup {self.id}: Merged polygon is invalid. Trying to fix it.")
                merged_polygon = merged_polygon.buffer(0)

            # if merged_polygon is MultiPolygon, take convex hull
            is_valid, status = evaluate_polygon(merged_polygon)
            if status == "multi_polygon":
                print(f"[CellGroup][WARN] CellGroup {self.id}: Merged polygon is MultiPolygon, taking convex hull.")
                merged_polygon = merged_polygon.convex_hull

            # check final validity
            is_valid, status = evaluate_polygon(merged_polygon)
            if not is_valid:
                print(f"[CellGroup][ERROR] CellGroup {self.id}: Merged polygon is still invalid after fixing. Status: {status}")

            self.merged_cell = Cell(self.id, merged_polygon, self.first_cell.event, resolution=self.first_cell.resolution)

        # Update the group index for all cells in the group
        for cell in self.cells:
            cell.set_group_id(self.id)
