from scripts.boustrophedon.cell import Cell, CellGroup
from scripts.boustrophedon.optimize_mcells_seq import OptimCellsSeq


def create_sequence(mcells: list[Cell]):
    opt_seq = OptimCellsSeq(mcells)
    return opt_seq.tour_ids


def set_merged_cells_adjacents(cells_groups: list[CellGroup], cells: list[Cell]):
    for cellg in cells_groups:
        parents_i = cellg.first_cell.mid_parentlist
        successors_i = cellg.last_cell.mid_adjlist
        cellg.merged_cell.mid_parentlist = [cells[i].group_id for i in parents_i]
        cellg.merged_cell.mid_adjlist = [cells[i].group_id for i in successors_i]
        adj = cellg.merged_cell.mid_parentlist.copy()
        adj.extend(cellg.merged_cell.mid_adjlist)
        cellg.merged_cell.adjacent_ids = adj


def create_cell_groups(cells):
    ncg = 0
    cells_groups: list[CellGroup] = []
    mcells: list[Cell] = []
    #
    for cell in cells[0:]:
        if cell.done:
            continue
        base_cell = cell
        base_cell.done = True
        if base_cell.n_adj == 0:
            new_group = CellGroup(base_cell, ncg)
        elif base_cell.n_adj > 1:
            new_group = CellGroup(base_cell, ncg)
        else:
            adj_cell: Cell = None
            if base_cell.up_adj_ci:
                adj_cell = cells[base_cell.up_adj_ci]
            elif base_cell.down_adj_ci:
                adj_cell = cells[base_cell.down_adj_ci]
            else:
                adj_cell = cells[base_cell.mid_adjlist[0]]

            #
            new_group = CellGroup(base_cell, ncg)
            while adj_cell.n_parent == 1 and not adj_cell.done:
                adj_cell.done = True
                new_group.add_cell(adj_cell)
                #
                if not adj_cell.n_adj == 1:
                    break
                #
                if adj_cell.up_adj_ci:
                    adj_cell = cells[adj_cell.up_adj_ci]
                elif adj_cell.down_adj_ci:
                    adj_cell = cells[adj_cell.down_adj_ci]
                else:
                    adj_cell = cells[adj_cell.mid_adjlist[0]]
        # merge_cells
        if new_group:
            new_group.update_group()
            cells_groups.append(new_group)
            mcells.append(new_group.merged_cell)
            ncg += 1
    #
    set_merged_cells_adjacents(cells_groups, cells)
    return cells_groups, mcells


class PostProcessMCells:
    def __init__(self, mcells):
        self.mcells = mcells

    def process(self):
        # Example processing: just return the data for now
        return self.mcell_data


def find_cells_shared_edge_p(cells: list[Cell]):
    P = 0.8
    for cell in cells:
        # right side
        p0_up = cell.ru
        p0_down = cell.rd
        l0 = p0_up[1] - p0_down[1]  # dy
        if l0 == 0:
            continue
        l_overlap = 0
        l_sum = 0
        inds = []
        for i, adj_i in enumerate(cell.mid_adjlist):
            adjc: Cell = cells[adj_i]
            p1_up = adjc.lu
            p1_down = adjc.ld
            l1 = p1_up[1] - p1_down[1]  # dy
            if l1 == 0:
                continue
            # find overlap
            overlap_min = max(p0_down[1], p1_down[1])
            overlap_max = min(p0_up[1], p1_up[1])
            l1_overlap = 0
            if overlap_max > overlap_min:
                l1_overlap = overlap_max - overlap_min
            # left side overlap of the adjacent cell
            p1 = l1_overlap/l1
            if p1 > P:
                l_overlap += l1_overlap
                l_sum += l1
                inds.append(i)
        # total overlap of the right edge of the cell
        p0 = l_overlap/l0
        if p0 < P:
            continue
        if l_sum > l0:
            cell.shift_re = True
        else:
            for i in inds:
                cells[cell.mid_adjlist[i]].shift_le = True

    # check parent cells
    for cell in cells:
        if cell.shift_le:
            continue
        # left side
        p0_up = cell.lu
        p0_down = cell.ld
        l0 = p0_up[1] - p0_down[1]  # dy
        if l0 == 0:
            continue
        l_overlap = 0
        l_sum = 0
        inds = []
        for i, adj_i in enumerate(cell.mid_parentlist):
            adjc: Cell = cells[adj_i]
            if adjc.shift_re:
                continue
            p1_up = adjc.ru
            p1_down = adjc.rd
            l1 = p1_up[1] - p1_down[1]  # dy
            if l1 == 0:
                continue
            # find overlap
            overlap_min = max(p0_down[1], p1_down[1])
            overlap_max = min(p0_up[1], p1_up[1])
            l1_overlap = 0
            if overlap_max > overlap_min:
                l1_overlap = overlap_max - overlap_min
            # right side overlap of the adjacent cell
            p1 = l1_overlap/l1
            if p1 > P:
                l_overlap += l1_overlap
                l_sum += l1
                inds.append(i)
        # total overlap of the left edge of the cell
        p0 = l_overlap/l0
        if p0 < P:
            continue
        if l_sum > l0:
            cell.shift_le = True
        else:
            for i in inds:
                cells[cell.mid_parentlist[i]].shift_re = True
