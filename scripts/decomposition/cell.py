from shapely import Polygon
from shapely.ops import unary_union
from scripts.decomposition.events import Event


class Cell:

    def __init__(self, ind: int, poly: Polygon, event: Event):
        self.ind: int = ind
        self.event: Event = event
        self.poly: Polygon = poly
        #
        self.xy: tuple[float, float] = None
        self.x_min: float = None
        self.x_max: float = None
        self.y_min: float = None
        self.y_max: float = None
        self.x_c: float = None
        self.y_c: float = None
        self.width: float = None
        self.height: float = None
        self.area: float = None
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
        self.done = False
        self.status = True
        #
        self.first_cell: Cell = None
        self.last_cell: Cell = None
        #
        self.group_ind: int = None

        # calculate cell specs
        self.cell_specs()

    def cell_specs(self):
        # center
        self.xy = self.poly.centroid.coords[0]
        # nounds
        minx, miny, maxx, maxy = self.poly.bounds
        self.x_min, self.y_min = minx, miny
        self.x_max, self.y_max = maxx, maxy
        # center
        self.x_c = (self.x_min + self.x_max) / 2
        self.y_c = (self.y_min + self.y_max) / 2
        # width and height
        self.width = maxx - minx
        self.height = maxy - miny
        # area
        self.area = self.poly.area

    def set_up_adj_ci(self, ci: str):
        if ci:
            self.up_adj_ci = ci
            self.n_adj += 1

    def set_down_adj_ci(self, ci: str):
        if ci:
            self.down_adj_ci = ci
            self.n_adj += 1

    def add_mid_adjlist(self, ci: str):
        if ci:
            self.mid_adjlist.append(ci)
            self.n_adj += 1

    def set_parent_up_ci(self, ci: str):
        if ci is not None:
            if self.parent_up_ci:
                self.add_mid_parentlist(ci)
            else:
                self.parent_up_ci = ci
                self.n_parent += 1

    def set_parent_down_ci(self, ci: str):
        if ci is not None:
            if self.parent_down_ci:
                self.add_mid_parentlist(ci)
            else:
                self.parent_down_ci = ci
                self.n_parent += 1

    def add_mid_parentlist(self, ci: str):
        if ci is not None:
            self.mid_parentlist.append(ci)
            self.n_parent += 1


class CellGroup:
    _counter = 0

    def __init__(self, first_cell: Cell, ncg: int = None):
        if ncg is None:
            self.ind = CellGroup._counter
        else:
            self.ind = ncg
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
