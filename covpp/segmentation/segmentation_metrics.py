""" Calculate properties of segmented cells. """
from dataclasses import dataclass
import numpy as np
from covpp.segmentation.cell_base import CellBase


@dataclass
class SegmentationMetrics:
    """Class to hold segmentation metrics.

    Attributes:
        average_area: Average area of segmented cells.
        average_perimeter: Average perimeter of segmented cells.
        average_free_perimeter: Average free perimeter of segmented cells.
        average_compactness_a: Average compactness (area / perimeter^2) of segmented cells.
        average_compactness_b: Average compactness (area / area of bounding box) of segmented cells.
    """
    average_area: float = None
    average_perimeter: float = None
    average_free_perimeter: float = None
    average_compactness_a: float = None
    average_compactness_b: float = None


class EvaluateSegmentationMetrics:
    def __init__(self, b_map: np.ndarray, cells: list[CellBase], gloabl_xy: np.ndarray):
        self.b_map = b_map
        self.cells = cells
        self.n_cells = len(cells)
        if self.n_cells == 0:
            raise ValueError("[SegmentationMetrics][ERROR] No cells provided for metrics calculation.")

        # Calculate the free perimeter for each cell based on the binary map
        for cell in self.cells:
            cell.calculate_free_perimeter(self.b_map, gloabl_xy)

        self.metrics = SegmentationMetrics(
            average_area=self.average_area(),
            average_perimeter=self.average_perimeter(),
            average_free_perimeter=self.average_free_perimeter(),
            average_compactness_a=self.average_compactness_a(),
            average_compactness_b=self.average_compactness_b()
        )

    def get_metrics(self) -> SegmentationMetrics:
        return self.metrics

    def average_area(self):
        total_area = sum(cell.area for cell in self.cells)
        return total_area / self.n_cells

    def average_perimeter(self):
        total_perimeter = sum(cell.perimeter for cell in self.cells)
        return total_perimeter / self.n_cells

    def average_free_perimeter(self):
        total_free_perimeter = sum(cell.free_perimeter for cell in self.cells)
        return total_free_perimeter / self.n_cells

    def average_compactness_a(self):
        total_compactness_a = sum(cell.compactness_a for cell in self.cells)
        return total_compactness_a / self.n_cells

    def average_compactness_b(self):
        total_compactness_b = sum(cell.compactness_b for cell in self.cells)
        return total_compactness_b / self.n_cells
