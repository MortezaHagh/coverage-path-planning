from shapely import Polygon
from shapely.ops import unary_union
from scripts.segmentation.cell_base import CellBase
from scripts.planning.boustrophedon.events import Event


class Cell (CellBase):
    """ Cell for boustrophedon decomposition. """

    def __init__(self, ind: int, poly: Polygon, event: Event):
        # initialize base class
        super().__init__(ind, poly)

        #
        self.group_id: int = None
        self.event: Event = event

        #
        self.ld: tuple[float, float] = None
        self.lu: tuple[float, float] = None
        self.rd: tuple[float, float] = None
        self.ru: tuple[float, float] = None
        #
        self.up_adj_ci = None
        self.down_adj_ci = None
        self.parent_up_ci = None
        self.parent_down_ci = None
        #
        self.n_adj = 0
        self.n_parent = 0
        #
        self.mid_adjlist = []
        self.mid_parentlist = []
        #
        self.grouped = False
        self.found_connection = True
        #
        # for cov planning
        self.shift_re: bool = False
        self.shift_le: bool = False

        # --------------------------------------

        # find corners
        self.find_corner()

    def set_group_id(self, group_id: int):
        if group_id is None:
            raise ValueError("Group ID cannot be None")
        self.group_id = group_id

    def find_corner(self):
        lps = [xy for xy in self.points if xy[0] == int(self.x_min)]  # todo toeval
        lps = sorted(lps, key=lambda xy: xy[1])
        self.ld = lps[0]
        self.lu = lps[-1]
        #
        rps = [xy for xy in self.points if xy[0] == int(self.x_max)]  # todo toeval
        rps = sorted(rps, key=lambda xy: xy[1])
        self.rd = rps[0]
        self.ru = rps[-1]

    def set_up_adj_ci(self, ci: int):
        if ci is not None:
            self.up_adj_ci = ci
            self.add_mid_adjlist(ci)

    def set_down_adj_ci(self, ci: int):
        if ci is not None:
            self.down_adj_ci = ci
            self.add_mid_adjlist(ci)

    def add_mid_adjlist(self, ci: int):
        if ci is not None:
            self.mid_adjlist.append(ci)
            self.n_adj += 1
            self.add_adjacent_id(ci)

    def set_parent_up_ci(self, ci: int):
        if ci is not None:
            if self.parent_up_ci is None:
                self.parent_up_ci = ci
            self.add_mid_parentlist(ci)

    def set_parent_down_ci(self, ci: int):
        if ci is not None:
            if self.parent_down_ci is None:
                self.parent_down_ci = ci
            self.add_mid_parentlist(ci)

    def add_mid_parentlist(self, ci: int):
        if ci is not None:
            self.mid_parentlist.append(ci)
            self.n_parent += 1
            self.add_adjacent_id(ci)

    def eval_planning_area(self, resolution, cleaning_r):
        min_area = 0.5 * 3.14 * cleaning_r**2
        if (self.area*resolution**2) < min_area:
            # print(f"[WARN] cell area is too small: {self.id}")
            self.do_planning = False

    def set_shift_re(self, shift: bool):
        self.shift_re = shift

    def set_shift_le(self, shift: bool):
        self.shift_le = shift


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
        self.cells: list[Cell] = [first_cell]
        self.first_cell: Cell = first_cell
        self.last_cell: Cell = None
        self.merged_cell: Cell = None

    def add_cell(self, cell: Cell):
        if cell:
            self.cells.append(cell)

    def set_first_cell(self, cell: 'Cell'):
        if cell:
            self.first_cell = cell

    def set_last_cell(self, cell: 'Cell'):
        if cell:
            self.last_cell = cell

    def update_group(self):
        if self.cells:
            self.set_last_cell(self.cells[-1])
            self.merge_cells()

    def merge_cells(self):
        # Merge all polygons
        if self.cells:
            merged_polygon = unary_union([cell.poly for cell in self.cells])
            self.merged_cell = Cell(self.id, merged_polygon, self.first_cell.event)

        # Update the group index for all cells in the group
        for cell in self.cells:
            cell.set_group_id(self.id)
