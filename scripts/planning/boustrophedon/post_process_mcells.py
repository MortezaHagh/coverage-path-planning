from scripts.planning.boustrophedon.cell import Cell
from scripts.planning.boustrophedon.optimize_mcells_seq import OptimCellsSeq


def create_sequence(mcells: list[Cell]):
    opt_seq = OptimCellsSeq(mcells)
    return opt_seq.tour_ids, opt_seq.cells_seq_length


def eval_cells_path_shift(cells: list[Cell]):
    prob_lim = 0.8
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
            if p1 > prob_lim:
                l_overlap += l1_overlap
                l_sum += l1
                inds.append(i)
        # total overlap of the right edge of the cell
        p0 = l_overlap/l0
        if p0 < prob_lim:
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
            if p1 > prob_lim:
                l_overlap += l1_overlap
                l_sum += l1
                inds.append(i)
        # total overlap of the left edge of the cell
        p0 = l_overlap/l0
        if p0 < prob_lim:
            continue
        if l_sum > l0:
            cell.shift_le = True
        else:
            for i in inds:
                cells[cell.mid_parentlist[i]].shift_re = True
