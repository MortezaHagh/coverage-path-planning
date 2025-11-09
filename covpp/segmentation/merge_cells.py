
""" Module to merge adjacent cells based on certain criteria. """
# from shapely import coverage_union
from shapely.ops import unary_union
from shapely.geometry import LineString
from covpp.segmentation.cell_base import CellBase
from covpp.base.logger import MyLogger


class MergeCells:
    """ Class to merge adjacent cells based on certain criteria. """

    def __init__(self, cells: list[CellBase]):
        self.lg = MyLogger("MergeCells")
        self.cells: list[CellBase] = cells
        self.updated_cells: list[CellBase] = []
        self.resolution = cells[0].resolution

        #
        group_cell_ids = self.find_merge_groups()
        merged_cells = self.merge_groups(group_cell_ids)
        self.updated_cells = merged_cells

    def find_merge_groups(self) -> list[set[int]]:
        skip_indices: set[int] = set()
        cell_id_map = {cell.id: cell for cell in self.cells}

        merge_groups: list[set[int]] = []

        # method:
        # 1. iterate through cells
        # 2. for each cell, like depth-first search, find all adjacent cells that can be merged
        for cell in self.cells:
            if cell.id in skip_indices:
                continue
            skip_indices.add(cell.id)

            group_cell_ids: set[int] = {cell.id}
            stack: list[int] = [cell.id]

            while stack:
                current_id = stack.pop()
                current_cell = cell_id_map[current_id]

                for j in current_cell.adjacent_ids:
                    other_cell = cell_id_map.get(j)
                    if other_cell.id in skip_indices:
                        continue
                    if self.should_merge(current_cell, other_cell):
                        group_cell_ids.add(other_cell.id)
                        skip_indices.add(other_cell.id)
                        stack.append(other_cell.id)

            merge_groups.append(group_cell_ids)
        return merge_groups

    def merge_groups(self, group_cell_ids: list[set[int]]) -> None:
        # if len(merge_group) <= 1: keep as is
        # else: merge all cells in the group and create a new cell
        merged_cells: list[CellBase] = []
        cell_id_map = {cell.id: cell for cell in self.cells}
        for group_ids in group_cell_ids:
            new_id = min(group_ids)
            if len(group_ids) <= 1:
                cell_id = group_ids.pop()
                merged_cells.append(cell_id_map[cell_id])
            else:
                self.lg.info(f"Merging group: {group_ids}")
                merged_polys = [cell_id_map[cell_id].base_poly for cell_id in group_ids]
                new_poly = unary_union(merged_polys)  # .convex_hull
                # new_poly = new_poly.convex_hull
                new_cell = CellBase(idx=new_id, poly=new_poly, resolution=self.resolution)
                # update adjacent ids
                merged_adjacent_ids = set()
                for cell_id in group_ids:
                    merged_adjacent_ids.update(cell_id_map[cell_id].adjacent_ids)
                merged_adjacent_ids.difference_update(group_ids)
                new_cell.adjacent_ids = list(merged_adjacent_ids)
                merged_cells.append(new_cell)
                # update adjacent ids of neighboring cells
                for adj_id in merged_adjacent_ids:
                    adj_cell = cell_id_map[adj_id]
                    adj_cell.adjacent_ids = [new_id if x in group_ids else x for x in adj_cell.adjacent_ids]
        return merged_cells

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
                    self.lg.info(f"Merging cell {cell.id} with cell {other_cell.id}")
                    cell = self.merge_two_cells(cell, other_cell)
                    skip_indices.add(other_cell.id)
                    break

            merged_cells.append(cell)
        return merged_cells

    def should_merge(self, cell1: CellBase, cell2: CellBase) -> bool:
        """ Determine if two cells should be merged based on criteria. """
        # Check if both cells are small
        if cell1.size_type == "small" and cell2.size_type == "small":
            return True

        # Check if both cells are slim
        if cell1.slim_type == "slim" and cell2.slim_type == "slim":
            if self.check_line_within_union(cell1, cell2):
                return True

        # check if either cell is slim or small
        checks = 0
        if cell1.slim_type == "slim" or cell1.size_type == "small":
            checks += 1
        if cell2.slim_type == "slim" or cell2.size_type == "small":
            checks += 1
        if checks == 2:
            if self.check_line_within_union(cell1, cell2):
                return True

        return False

    def check_line_within_union(self, cell1: CellBase, cell2: CellBase) -> bool:
        """ Check if the line connecting the centroids of two cells lies within their union polygon. """

        centroid1 = cell1.base_poly.centroid
        centroid2 = cell2.base_poly.centroid
        line = LineString([centroid1, centroid2])
        union_poly = unary_union([cell1.base_poly, cell2.base_poly])
        # union_poly = union_poly.convex_hull
        return union_poly.contains(line)

    def merge_two_cells(self, cell1: CellBase, cell2: CellBase) -> CellBase:
        """ Merge two cells into one, normalizing MultiPolygon to a single Polygon. """
        # new_poly = coverage_union(cell1.poly, cell2.poly).normalize()
        new_poly = unary_union([cell1.base_poly, cell2.base_poly])
        # new_poly = new_poly.convex_hull
        new_cell = CellBase(idx=cell1.id, poly=new_poly, resolution=cell1.resolution)
        merged_adjacent_ids = set(cell1.adjacent_ids).union(set(cell2.adjacent_ids))
        merged_adjacent_ids.discard(cell1.id)
        merged_adjacent_ids.discard(cell2.id)
        new_cell.adjacent_ids = list(merged_adjacent_ids)

        # # plot for debugging
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # # fix aspect ratio
        # ax.set_aspect('equal', 'box')
        # x1, y1 = cell1.base_poly.exterior.xy
        # ax.fill(x1, y1, alpha=0.5, fc='red', ec='black', label='Cell 1')
        # x2, y2 = cell2.base_poly.exterior.xy
        # ax.fill(x2, y2, alpha=0.5, fc='blue', ec='black', label='Cell 2')
        # x_new, y_new = new_poly.exterior.xy
        # ax.fill(x_new, y_new, alpha=0.5, fc='green', ec='black', label='Merged Cell')
        # ax.legend()
        # plt.show()
        return new_cell
