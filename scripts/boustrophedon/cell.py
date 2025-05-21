from shapely import Polygon
from shapely.ops import unary_union
from scripts.boustrophedon.events import Event


class Cell:

    def __init__(self, ind: int, poly: Polygon, event: Event):
        self.ind: int = ind
        self.opt_ind: int = None
        self.next_cell_seq: int = None
        #
        self.event: Event = event
        self.poly: Polygon = poly
        self.points: list[tuple[int, int]] = []
        #
        self.cxy: tuple[float, float] = None
        self.x_min: float = None
        self.x_max: float = None
        self.y_min: float = None
        self.y_max: float = None
        self.x_mid: float = None
        self.y_mid: float = None
        self.width: float = None
        self.height: float = None
        self.area: float = None
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
        self.adjacent_inds: list[int] = []
        #
        self.done = False
        self.status = True
        #
        self.first_cell: Cell = None
        self.last_cell: Cell = None
        #
        self.group_ind: int = None
        # for cov planning
        self.shift_re: bool = False
        self.shift_le: bool = False

        # --------------------------------------

        # calculate cell specs
        self.cell_specs()

        # create cell points
        self.create_cell_points()

        # find corners
        self.find_corner()

    def cell_specs(self):
        # center
        self.cxy = self.poly.centroid.coords[0]
        # nounds
        minx, miny, maxx, maxy = self.poly.bounds
        self.x_min, self.y_min = minx, miny
        self.x_max, self.y_max = maxx, maxy
        # center
        self.x_mid = (self.x_min + self.x_max) / 2
        self.y_mid = (self.y_min + self.y_max) / 2
        # width and height
        self.width = maxx - minx
        self.height = maxy - miny
        # area
        self.area = self.poly.area

    def create_cell_points(self):
        # create cell points
        if self.poly:
            self.points = [(int(p[0]), int(p[1])) for p in self.poly.exterior.coords]
        else:
            self.points = []

    def find_corner(self):
        lps = [xy for xy in self.points if xy[0] == self.x_min]
        lps = sorted(lps, key=lambda xy: xy[1])
        self.ld = lps[0]
        self.lu = lps[-1]
        #
        rps = [xy for xy in self.points if xy[0] == self.x_max]
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
            self.add_adjacent_ind(ci)

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
            self.add_adjacent_ind(ci)

    def add_adjacent_ind(self, ind: int):
        self.adjacent_inds.append(ind)

    def set_optim_ind(self, ind: int, next_cell_ind: int = None):
        self.opt_ind = ind
        if next_cell_ind is not None:
            self.next_cell_seq = next_cell_ind
        else:
            self.next_cell_seq = None


class CellGroup:
    _counter = 0

    def __init__(self, first_cell: Cell, ncg: int = None):
        if ncg is None:
            self.ind = CellGroup._counter
            self.opt_ind = CellGroup._counter
        else:
            self.ind = ncg
            self.opt_ind = ncg
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
            self.last_cell = self.cells[-1]
            self.merge_cells()

    def merge_cells(self):
        # Merge all polygons
        if self.cells:
            merged_polygon = unary_union([cell.poly for cell in self.cells])
            self.merged_cell = Cell(self.ind, merged_polygon, self.first_cell.event)

        # Update the group index for all cells in the group
        for cell in self.cells:
            cell.group_ind = self.ind
