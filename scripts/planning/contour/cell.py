import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely import Polygon, maximum_inscribed_circle, minimum_bounding_circle
from scripts.base.logger import MyLogger
from scripts.planning.base_cell import BaseCell
from scripts.base.configs import PlanningConfigs
from scripts.planning.contour.common import CriticalPoint


class Cell (BaseCell):
    basic: bool  # whether the cell is basic or not
    critical_points: list[CriticalPoint]  # critical points of the cell
    rect_circ_like: bool  # whether the cell is rectangle-like or not
    min_clearance: float  # minimum clearance of the cell
    clearance_regular: bool  # whether the cell is regular or not
    mic_cell_area_regular: bool  # maximum inscribed circle area ratio to cell area
    cell_mbc_area_regular: bool  # minimum bounding circle area ratio to cell area

    def __init__(self, ind: int, poly: Polygon, resolution: float = None) -> None:
        # initialize base class
        super().__init__(ind, poly)
        self.ns = f"Cell-{self.id}"
        self.lg = MyLogger(self.ns)

        # settings
        if resolution is None:
            raise ValueError("Resolution must be provided for the cell.")
        self.robot_r_px = int(PlanningConfigs.cleaning_r_ov / resolution)

        #
        self.basic = False
        self.critical_points = []
        #
        self.rect_circ_like = False
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

        poly_area = self.poly.area

        # minimum_rotated_rectangle polygon
        rect_circ_poly = self.poly.minimum_rotated_rectangle
        rect_circ_area = rect_circ_poly.area

        # maximum_inscribed_circle
        circle_insc_poly = maximum_inscribed_circle(self.poly)
        circle_insc_center, circle_insc_boundary = circle_insc_poly.coords
        circle_insc_radius = np.linalg.norm(np.array(circle_insc_center) - np.array(circle_insc_boundary))
        circle_insc_area = np.pi * circle_insc_radius ** 2

        # minimum circumscribed circle
        circle_circ_poly = minimum_bounding_circle(self.poly)
        circle_circ_area = circle_circ_poly.area

        # # eval regularities
        self.eval_rectangle_like_circumscribed(rect_circ_area, poly_area)
        inscribed_rect_poly = self.create_inscribed_rectangle(rect_circ_poly, circle_insc_center, circle_insc_radius)
        expand_inscribed_rectangle = self.expand_inscribed_rectangle(inscribed_rect_poly, circle_insc_center)
        print(" --------------------- ")

        #
        self.plot_cell_evaluation(
            circle_insc_center=circle_insc_center,
            circle_insc_radius=circle_insc_radius,
            circle_circ_poly=circle_circ_poly,
            rect_circ_poly=rect_circ_poly,
            inscribed_rect_poly=inscribed_rect_poly,
            expand_inscribed_rectangle=expand_inscribed_rectangle
        )

    def eval_rectangle_like_circumscribed(self, rect_circ_area, poly_area) -> None:
        # based on inscribed and circumscribed rectangles
        cell_rect_circ_area_ration = poly_area/rect_circ_area
        self.rect_circ_like = cell_rect_circ_area_ration >= 0.8
        print(f"cell/rect_circ area: {cell_rect_circ_area_ration:.2f}, {self.rect_circ_like}")

    def create_inscribed_rectangle(self, rect_circ_poly, circle_insc_center, circle_insc_radius) -> Polygon:
        # calculate the angle of the rectangle
        rect_circ_coords = np.array(rect_circ_poly.exterior.coords)
        angle = np.arctan2(rect_circ_coords[1, 1] - rect_circ_coords[0, 1],
                           rect_circ_coords[1, 0] - rect_circ_coords[0, 0])

        # create initial rectangle inside the inscribed circle
        rect_coords = []
        theta = angle + np.pi / 4  # 45 degrees rotation for the rectangle
        for i in range(4):
            angle_offset = theta + i * np.pi / 2
            x = circle_insc_center[0] + circle_insc_radius * np.cos(angle_offset)
            y = circle_insc_center[1] + circle_insc_radius * np.sin(angle_offset)
            rect_coords.append((x, y))
        # create the rectangle polygon
        inscribed_rect_poly = Polygon(rect_coords)
        return inscribed_rect_poly

    def expand_inscribed_rectangle(self, inscribed_rect_poly: Polygon, circle_insc_center) -> Polygon:
        step = self.robot_r_px
        min_step = step / 5
        #
        coords = inscribed_rect_poly.exterior.coords[:-1]
        coords = list(coords)
        idx1s = [0, 1, 2, 3]
        idx2s = [1, 2, 3, 0]
        steps = [step for _ in range(len(idx2s))]
        last_try = [False for _ in range(len(idx2s))]

        # calculate the angles of the midpoints of the edges
        thetas = []
        for idx1, idx2 in zip(idx1s, idx2s):
            p1 = coords[idx1]
            p2 = coords[idx2]
            midp = np.mean([p1, p2], axis=0)
            th = np.arctan2(midp[1] - circle_insc_center[1], midp[0] - circle_insc_center[0])
            thetas.append(th)

        # expand seuentially
        any_valid = True
        while any_valid:
            any_valid = False
            for i, (idx1, idx2, th) in enumerate(zip(idx1s, idx2s, thetas)):
                step = steps[i]
                p1 = coords[idx1]
                p2 = coords[idx2]
                midp = np.mean([p1, p2], axis=0)
                # shift p1 and p2 in the theta_midp_c direction
                new_p1 = (
                    p1[0] + step * np.cos(th),
                    p1[1] + step * np.sin(th)
                )
                new_p2 = (
                    p2[0] + step * np.cos(th),
                    p2[1] + step * np.sin(th)
                )
                # check if the new points are inside the polygon
                is_p1_in = self.poly.contains(Point(new_p1))
                is_p2_in = self.poly.contains(Point(new_p2))
                if is_p1_in and is_p2_in:
                    any_valid = True
                    coords[idx1] = new_p1
                    coords[idx2] = new_p2
                elif steps[i] > min_step:
                    any_valid = True
                    steps[i] = step / 2  # reduce the step size
                elif not last_try[i]:
                    last_try[i] = True
                    new_p1 = (
                        p1[0] - step * np.cos(th),
                        p1[1] - step * np.sin(th)
                    )
                    new_p2 = (
                        p2[0] - step * np.cos(th),
                        p2[1] - step * np.sin(th)
                    )
                    any_valid = True
                    coords[idx1] = new_p1
                    coords[idx2] = new_p2
                else:
                    pass

        # create the expanded rectangle polygon
        expanded_rect_poly = Polygon(coords)
        return expanded_rect_poly

    # def eval_min_side_clearance_ratio(self, rect_circ_poly: Polygon) -> None:
    #     mrr_coords = np.array(rect_circ_poly.exterior.coords)
    #     mrr_side_lengths = np.linalg.norm(np.diff(mrr_coords, axis=0), axis=1)
    #     mrr_min_side_length = np.min(mrr_side_lengths)
    #     s_ratio = self.min_clearance/mrr_min_side_length
    #     if s_ratio > 0.7:
    #         self.clearance_regular = True
    #     print(f"s_ratio: {s_ratio:.2f}, {self.clearance_regular}")

    # def eval_mic_cell_area_ratio(self, mic_area) -> None:
    #     mic_cell_area_ration = mic_area / self.poly.area
    #     self.mic_cell_area_regular = mic_cell_area_ration >= 0.8
    #     print(f"MIC area ratio: {mic_cell_area_ration:.2f}, {self.mic_cell_area_regular}")

    # def eval_cell_mrr_area_ratio(self, mrr_area) -> None:
    #     cell_mrr_area_ration = self.poly.area / mrr_area
    #     self.cell_mrr_area_regular = cell_mrr_area_ration >= 0.7
    #     print(f"MBC area ratio: {cell_mrr_area_ration:.2f}, {self.cell_mrr_area_regular}")

    def plot_cell_evaluation(self, circle_insc_center, circle_insc_radius, circle_circ_poly, rect_circ_poly, inscribed_rect_poly, expand_inscribed_rectangle) -> None:

        # plotting
        fig, ax = plt.subplots()
        # plot polygon
        x, y = self.poly.exterior.xy
        ax.plot(x, y, color='blue', label='Polygon')
        # plot centroid
        centroid = self.poly.centroid
        ax.plot(centroid.x, centroid.y, 'ro', label='Centroid')
        # plot mic
        ax.plot(circle_insc_center[0], circle_insc_center[1], 'go', label='MIC Center')
        circle = plt.Circle(circle_insc_center, circle_insc_radius, color='green', fill=False, label='MIC Radius')
        ax.add_artist(circle)
        # plot mbc
        mbc_x, mbc_y = circle_circ_poly.exterior.xy
        ax.plot(mbc_x, mbc_y, color='orange', label='Minimum Bounding Circle')
        # plot mrr
        mrr_x, mrr_y = rect_circ_poly.exterior.xy
        ax.plot(mrr_x, mrr_y, color='purple', label='Minimum Rotated Rectangle')

        # inscribed_rect_poly
        inscribed_rect_x, inscribed_rect_y = inscribed_rect_poly.exterior.xy
        ax.plot(inscribed_rect_x, inscribed_rect_y, color='cyan', label='Inscribed Rectangle')

        # expand_inscribed_rectangle
        expand_rect_x, expand_rect_y = expand_inscribed_rectangle.exterior.xy
        ax.plot(expand_rect_x, expand_rect_y, color='magenta', label='Expanded Inscribed Rectangle')

        #
        ax.set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title(f"Cell {self.id} Evaluation")
        plt.show()
