import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, maximum_inscribed_circle, minimum_bounding_circle
from scripts.base.logger import MyLogger
from scripts.planning.base_cell import BaseCell
from scripts.planning.contour.common import CriticalPoint


class Cell (BaseCell):
    basic: bool  # whether the cell is basic or not
    critical_points: list[CriticalPoint]  # critical points of the cell
    min_clearance: float  # minimum clearance of the cell
    clearance_regular: bool  # whether the cell is regular or not
    mic_cell_area_regular: bool  # maximum inscribed circle area ratio to cell area
    cell_mbc_area_regular: bool  # minimum bounding circle area ratio to cell area

    def __init__(self, ind: int, poly: Polygon):
        # initialize base class
        super().__init__(ind, poly)
        self.ns = f"Cell-{self.id}"
        self.lg = MyLogger(self.ns)

        #
        self.basic = False
        self.critical_points = []
        #
        self.min_clearance = 0.0
        self.clearance_regular = False
        self.mic_cell_area_regular = False
        self.cell_mrr_area_regular = False

    def set_critical_points(self, critical_points: list[CriticalPoint]) -> None:
        self.critical_points = critical_points
        self.find_min_clearance()

    def find_min_clearance(self) -> None:
        if not self.critical_points:
            self.basic = True
            self.lg.warn("No critical points set for the cell, marking as basic.")
            # raise ValueError("Critical points are not set for the cell.")
            return

        cp: CriticalPoint
        clearances = [cp.vp.clearance for cp in self.critical_points]
        self.min_clearance = min(clearances)

    def eval_cell(self):
        print(f"Cell {self.id} evaluation:")

        # minimum_rotated_rectangle polygon
        mrr_poly = self.poly.minimum_rotated_rectangle
        mrr_area = mrr_poly.area

        # maximum_inscribed_circle
        mic_poly = maximum_inscribed_circle(self.poly)
        mic_center, mic_boundary = mic_poly.coords
        mic_radius = np.linalg.norm(np.array(mic_center) - np.array(mic_boundary))
        mic_area = np.pi * mic_radius ** 2

        # minimum_bounding_circle
        mbc_poly = minimum_bounding_circle(self.poly)
        mbc_area = mbc_poly.area

        # eval regularities
        self.eval_min_side_clearance_ratio(mrr_poly)
        # self.eval_mic_cell_area_ratio(mic_area)
        self.eval_cell_mrr_area_ratio(mrr_area)
        # is_regular = self.clearance_regular or self.mic_cell_area_regular
        # print("Cell regularity:", is_regular)
        print(" --------------------- ")

        #
        self.plot_cell_evaluation(
            mic_center=mic_center,
            mic_radius=mic_radius,
            mbc_poly=mbc_poly,
            mrr_poly=mrr_poly
        )

    def eval_min_side_clearance_ratio(self, mrr_poly: Polygon) -> None:
        mrr_coords = np.array(mrr_poly.exterior.coords)
        mrr_side_lengths = np.linalg.norm(np.diff(mrr_coords, axis=0), axis=1)
        mrr_min_side_length = np.min(mrr_side_lengths)
        s_ratio = self.min_clearance/mrr_min_side_length
        if s_ratio > 0.7:
            self.clearance_regular = True
        print(f"s_ratio: {s_ratio:.2f}, {self.clearance_regular}")

    def eval_mic_cell_area_ratio(self, mic_area) -> None:
        mic_cell_area_ration = mic_area / self.poly.area
        self.mic_cell_area_regular = mic_cell_area_ration >= 0.8
        print(f"MIC area ratio: {mic_cell_area_ration:.2f}, {self.mic_cell_area_regular}")

    def eval_cell_mrr_area_ratio(self, mrr_area) -> None:
        cell_mrr_area_ration = self.poly.area / mrr_area
        self.cell_mrr_area_regular = cell_mrr_area_ration >= 0.7
        print(f"MBC area ratio: {cell_mrr_area_ration:.2f}, {self.cell_mrr_area_regular}")

    def plot_cell_evaluation(self, mic_center, mic_radius, mbc_poly, mrr_poly) -> None:

        # plotting
        fig, ax = plt.subplots()
        # plot polygon
        x, y = self.poly.exterior.xy
        ax.plot(x, y, color='blue', label='Polygon')
        # plot centroid
        centroid = self.poly.centroid
        ax.plot(centroid.x, centroid.y, 'ro', label='Centroid')
        # plot mic
        ax.plot(mic_center[0], mic_center[1], 'go', label='MIC Center')
        circle = plt.Circle(mic_center, mic_radius, color='green', fill=False, label='MIC Radius')
        ax.add_artist(circle)
        # plot mbc
        mbc_x, mbc_y = mbc_poly.exterior.xy
        ax.plot(mbc_x, mbc_y, color='orange', label='Minimum Bounding Circle')
        # plot mrr
        mrr_x, mrr_y = mrr_poly.exterior.xy
        ax.plot(mrr_x, mrr_y, color='purple', label='Minimum Rotated Rectangle')
        ax.set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title(f"Cell {self.id} Evaluation")
        plt.show()
