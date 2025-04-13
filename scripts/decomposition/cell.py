from shapely import Polygon
from scripts.decomposition.events import Event


class Cell:

    def __init__(self, ind: int, poly: Polygon, event: Event):
        self.ind: int = ind
        self.event: Event = event
        self.poly: Polygon = poly
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
        if ci:
            self.mid_parentlist.append(ci)
            self.n_parent += 1


class CellGroup:
    def __init__(self, first_cell: Cell):
        self.cells: list[Cell] = [first_cell]
        self.first_cell: Cell = first_cell
        self.last_cell: Cell = None

    def add_cell(self, cell: Cell):
        if cell:
            self.cells.append(cell)

    def set_first_cell(self, cell: 'Cell'):
        if cell:
            self.first_cell = cell

    def set_last_cell(self, cell: 'Cell'):
        if cell:
            self.last_cell = cell
