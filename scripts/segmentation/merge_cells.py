
# from shapely import coverage_union
from shapely.ops import unary_union
from scripts.segmentation.cell_base import CellBase


class MergeCells:
    """ Class to merge adjacent cells based on certain criteria. """

    def __init__(self, cells: list[CellBase]):
        self.cells: list[CellBase] = cells
        self.updated_cells: list[CellBase] = []
        self.merge_adjacent_cells()

    def merge_adjacent_cells(self) -> None:
        """ Merge adjacent cells if they meet the merging criteria. """
        merged_cells: list[CellBase] = []
        skip_indices: set[int] = set()
        cell_id_map = {cell.id: cell for cell in self.cells}

        for cell in self.cells:
            if cell.id in skip_indices:
                continue
            skip_indices.add(cell.id)

            for j in cell.adjacent_ids:
                other_cell = cell_id_map.get(j)
                if other_cell.id in skip_indices:
                    continue
                if self.should_merge(cell, other_cell):
                    print(f"Merging cell {cell.id} with cell {other_cell.id}")
                    cell = self.merge_two_cells(cell, other_cell)
                    skip_indices.add(other_cell.id)
                    break

            merged_cells.append(cell)

        self.updated_cells = merged_cells

    def should_merge(self, cell1: CellBase, cell2: CellBase) -> bool:
        """ Determine if two cells should be merged based on criteria. """
        if cell1.size_type == "small" and cell2.size_type == "small":
            return True
        return False

    def merge_two_cells(self, cell1: CellBase, cell2: CellBase) -> CellBase:
        """ Merge two cells into one, normalizing MultiPolygon to a single Polygon. """
        # new_poly = coverage_union(cell1.poly, cell2.poly).normalize()
        new_poly = unary_union([cell1.base_poly, cell2.base_poly])
        new_poly = new_poly.convex_hull
        new_cell = CellBase(idx=cell1.id, poly=new_poly, resolution=cell1.resolution)
        merged_adjacent_ids = set(cell1.adjacent_ids).union(set(cell2.adjacent_ids))
        merged_adjacent_ids.discard(cell1.id)
        merged_adjacent_ids.discard(cell2.id)
        new_cell.adjacent_ids = list(merged_adjacent_ids)

        # plot for debugging
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # fix aspect ratio
        ax.set_aspect('equal', 'box')
        x1, y1 = cell1.poly.exterior.xy
        ax.fill(x1, y1, alpha=0.5, fc='red', ec='black', label='Cell 1')
        x2, y2 = cell2.poly.exterior.xy
        ax.fill(x2, y2, alpha=0.5, fc='blue', ec='black', label='Cell 2')
        x_new, y_new = new_poly.exterior.xy
        ax.fill(x_new, y_new, alpha=0.5, fc='green', ec='black', label='Merged Cell')
        ax.legend()
        plt.show()
        return new_cell
