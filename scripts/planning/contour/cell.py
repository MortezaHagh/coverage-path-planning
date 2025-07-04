import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely import Polygon
from shapely.affinity import scale
from shapely import maximum_inscribed_circle, minimum_bounding_circle
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

        poly_area = self.area

        # minimum_rotated_rectangle polygon
        rect_circ_poly = self.poly.minimum_rotated_rectangle
        rect_circ_area = rect_circ_poly.area

        # maximum_inscribed_circle
        circle_insc_poly = maximum_inscribed_circle(self.poly)
        circle_insc_center, circle_insc_boundary = circle_insc_poly.coords
        circle_insc_radius = np.linalg.norm(np.array(circle_insc_center) - np.array(circle_insc_boundary))
        # circle_insc_area = np.pi * circle_insc_radius ** 2

        # # minimum circumscribed circle
        # circle_circ_poly = minimum_bounding_circle(self.poly)
        # circle_circ_area = circle_circ_poly.area

        # # eval regularities
        self.eval_rectangle_like_circumscribed(rect_circ_area, poly_area)
        inscribed_rect_poly = self.create_inscribed_rectangle(rect_circ_poly, circle_insc_center, circle_insc_radius)
        expanded_inscribed_rectangle = self.expand_inscribed_rectangle(inscribed_rect_poly, circle_insc_center)
        print(f"cell/rect_circ area: {self.rect_circ_like}")
        print(" --------------------- ")

        #
        # self.plot_cell_evaluation(
        #     rect_circ_poly=rect_circ_poly,
        #     inscribed_rect_poly=inscribed_rect_poly,
        #     expand_inscribed_rectangle=expanded_inscribed_rectangle,
        #     circle_insc_center=circle_insc_center,
        #     circle_insc_radius=circle_insc_radius,
        #     # circle_circ_poly=circle_circ_poly,
        # )

    def eval_rectangle_like_circumscribed(self, rect_circ_area, poly_area) -> None:
        k1 = 0.8
        # based on inscribed and circumscribed rectangles
        cell_rect_circ_area_ratio = poly_area/rect_circ_area
        self.rect_circ_like = cell_rect_circ_area_ratio >= k1

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
        k1 = 0.8
        # first expansion
        expanded_rect_poly_1 = self.expand_inscribed_rectangle_1(inscribed_rect_poly, circle_insc_center)
        area_1 = expanded_rect_poly_1.area
        rect_cell_a_r = area_1 / self.area
        print(f"r1: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > k1:
            self.rect_circ_like = True
            return expanded_rect_poly_1

        # second expansion
        expanded_rect_poly_2 = self.expand_inscribed_rectangle_2(inscribed_rect_poly, circle_insc_center)
        area_2 = expanded_rect_poly_2.area
        rect_cell_a_r = area_2 / self.area
        print(f"r2: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > k1:
            self.rect_circ_like = True
            return expanded_rect_poly_2

        # third expansion
        expanded_rect_poly_3 = self.expand_inscribed_rectangle_3()
        area_3 = expanded_rect_poly_3.area
        rect_cell_a_r = area_3 / self.area
        print(f"r3: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > k1:
            self.rect_circ_like = True
            return expanded_rect_poly_3

        #
        all_polys = [expanded_rect_poly_1, expanded_rect_poly_2, expanded_rect_poly_3]
        areas = [area_1, area_2, area_3]
        best_poly = max(zip(all_polys, areas), key=lambda x: x[1])[0]
        return best_poly

    def expand_inscribed_rectangle_1(self, inscribed_rect_poly: Polygon, circle_insc_center) -> Polygon:
        """ expand in 4 directions """

        # find step
        step = self.robot_r_px
        min_step = step / 5

        # scale inscribed_rect_poly
        sf = 0.8
        inscribed_rect_poly = scale(inscribed_rect_poly, xfact=sf, yfact=sf, origin=circle_insc_center)

        #
        coords = inscribed_rect_poly.exterior.coords[:-1]
        coords = list(coords)
        idx1s = [0, 1, 2, 3]
        idx2s = [1, 2, 3, 0]
        steps = [step for _ in range(len(idx2s))]
        # last_try = [False for _ in range(len(idx2s))]

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
                success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                if success:
                    any_valid = True
                    coords[idx1] = new_p1
                    coords[idx2] = new_p2
                elif steps[i] > min_step:
                    any_valid = True
                    steps[i] = step / 2  # reduce the step size
                # elif not last_try[i]:
                #     any_valid = True
                #     last_try[i] = True
                #     success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                #     coords[idx1] = new_p1
                #     coords[idx2] = new_p2

        # create the expanded rectangle polygon
        expanded_rect_poly = Polygon(coords)
        return expanded_rect_poly

    def expand_inscribed_rectangle_2(self, inscribed_rect_poly: Polygon, circle_insc_center) -> Polygon:
        """ first find the main direction of expansion, then expand in that direction and then in the other 3 directions """

        # find step
        # smallest length of the rectangle side
        rect_circ_coords = np.array(inscribed_rect_poly.exterior.coords)
        rect_circ_side_lengths = np.linalg.norm(np.diff(rect_circ_coords, axis=0), axis=1)
        min_rect_circ_side_length = np.min(rect_circ_side_lengths)
        step = min_rect_circ_side_length / 10
        # step = self.robot_r_px
        min_step = step / 5

        # scale inscribed_rect_poly
        sf = 0.3
        s_rect_poly = scale(inscribed_rect_poly, xfact=sf, yfact=sf, origin=circle_insc_center)

        #
        coords = s_rect_poly.exterior.coords[:-1]
        coords = list(coords)
        idx1s = [0, 1, 2, 3]
        idx2s = [1, 2, 3, 0]
        steps = [step for _ in range(4)]
        # last_try = [False for _ in range(len(idx2s))]

        # calculate the angles of the midpoints of the edges
        thetas = []
        for idx1, idx2 in zip(idx1s, idx2s):
            p1 = coords[idx1]
            p2 = coords[idx2]
            midp = np.mean([p1, p2], axis=0)
            th = np.arctan2(midp[1] - circle_insc_center[1], midp[0] - circle_insc_center[0])
            thetas.append(th)

        # find main direction of expansion
        # expand seperately in each direction till collision
        coords_cand = [coords.copy() for _ in range(4)]
        for i, (idx1, idx2, th) in enumerate(zip(idx1s, idx2s, thetas)):
            is_valid = True
            while is_valid:
                step = steps[i]
                p1 = coords_cand[i][idx1]
                p2 = coords_cand[i][idx2]
                midp = np.mean([p1, p2], axis=0)
                success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                if success:
                    coords_cand[i][idx1] = new_p1
                    coords_cand[i][idx2] = new_p2
                else:
                    is_valid = False

        # find the best candidate
        best_area = 0.0
        best_coords = None
        for coords_cand_i in coords_cand:
            expanded_rect_poly_cand = Polygon(coords_cand_i)
            area = expanded_rect_poly_cand.area
            if area > best_area:
                best_area = area
                best_coords = coords_cand_i

        # normal expansion in the all directions
        # expand seqentially
        any_valid = True
        for i, (idx1, idx2, th) in enumerate(zip(idx1s, idx2s, thetas)):
            any_valid = True
            while any_valid:
                step = steps[i]
                p1 = best_coords[idx1]
                p2 = best_coords[idx2]
                midp = np.mean([p1, p2], axis=0)
                success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                if success:
                    any_valid = True
                    best_coords[idx1] = new_p1
                    best_coords[idx2] = new_p2
                elif steps[i] > min_step:
                    any_valid = True
                    steps[i] = step / 2  # reduce the step size
                else:
                    any_valid = False

        # create the expanded rectangle polygon
        expanded_rect_poly = Polygon(best_coords)
        return expanded_rect_poly

    def expand_inscribed_rectangle_3(self) -> Polygon:
        """ expand based on the biggest edge """

        # step
        step = self.robot_r_px
        min_step = step / 3

        # find biggest edge of the self.poly
        coords = np.array(self.poly.exterior.coords)
        edges = np.diff(coords, axis=0)
        edge_lengths = np.linalg.norm(edges, axis=1)
        max_edge_idx = np.argmax(edge_lengths)

        # twp edge points
        p1 = coords[max_edge_idx]
        p2 = coords[(max_edge_idx + 1) % len(coords)]

        # calculate the angle of the edge p1->p2
        th = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

        #  move p1 and p2 in the direction of the center of the edge
        l = max_edge_idx / 4
        s_p1 = (p1[0] + l * np.cos(th),
                p1[1] + l * np.sin(th))
        s_p2 = (p2[0] + l * np.cos(-th),
                p2[1] + l * np.sin(-th))

        # create third and fourth points
        # p3 is the point from p2 perpendicular to edge p1-p2
        th3 = np.pi / 2 + th,
        p3 = (
            s_p2[0] + step * np.cos(th3),
            s_p2[1] + step * np.sin(th3)
        )
        if not self.poly.contains(Point(p3)):
            th3 = th - np.pi / 2
            p3 = (
                s_p1[0] + step * np.cos(th3),
                s_p1[1] + step * np.sin(th3)
            )
        if not self.poly.contains(Point(p3)):
            raise ValueError("Cannot find a valid point p3 for the inscribed rectangle expansion.")

        # move p1 and p2 alittle inside
        delta = step/5
        p1 = (s_p1[0] + delta * np.cos(th3),
              s_p1[1] + delta * np.sin(th3))
        p2 = (s_p2[0] + delta * np.cos(th3),
              s_p2[1] + delta * np.sin(th3))

        # p4 is the point from p1 perpendicular to edge p1-p2
        p4 = (s_p1[0] + step * np.cos(th3),
              s_p1[1] + step * np.sin(th3))
        if not self.poly.contains(Point(p4)):
            raise ValueError("Cannot find a valid point p4 for the inscribed rectangle expansion.")

        # create new polygon
        coords = [p1, p2, p3, p4]  # new polygon
        new_poly = Polygon(coords)
        center = new_poly.centroid.coords[0]

        #
        coords = list(coords)
        idx1s = [1, 2, 3]
        idx2s = [2, 3, 0]
        steps = [step for _ in range(3)]

        # calculate the angles of the midpoints of the edges
        thetas = []
        for idx1, idx2 in zip(idx1s, idx2s):
            p1 = coords[idx1]
            p2 = coords[idx2]
            midp = np.mean([p1, p2], axis=0)
            th = np.arctan2(midp[1] - center[1], midp[0] - center[0])
            thetas.append(th)

        # expand seqentially
        any_valid = True
        for i, (idx1, idx2, th) in enumerate(zip(idx1s, idx2s, thetas)):
            any_valid = True
            while any_valid:
                step = steps[i]
                p1 = coords[idx1]
                p2 = coords[idx2]
                midp = np.mean([p1, p2], axis=0)
                success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                if success:
                    any_valid = True
                    coords[idx1] = new_p1
                    coords[idx2] = new_p2
                elif steps[i] > min_step:
                    any_valid = True
                    steps[i] = step / 2  # reduce the step size
                else:
                    any_valid = False

        # create the expanded rectangle polygon
        expanded_rect_poly = Polygon(coords)

        #
        fig, ax = plt.subplots()
        # main polygon
        x, y = self.poly.exterior.xy
        ax.plot(x, y, 'o-', color='blue', label='Polygon')
        # rect circumscribed polygon
        x, y = new_poly.exterior.xy
        ax.plot(x, y, color='purple', label='new_poly')
        #
        x, y = expanded_rect_poly.exterior.xy
        ax.plot(x, y, color='magenta', label='Expanded Rectangle')
        ax.set_aspect('equal', adjustable='box')
        #
        plt.legend()
        plt.show()

        return expanded_rect_poly

    def expand_2_points_eval(self, p1, p2, th, step):
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
        success = False
        is_p1_in = self.poly.contains(Point(new_p1))
        is_p2_in = self.poly.contains(Point(new_p2))
        if is_p1_in and is_p2_in:
            success = True
        return success, new_p1, new_p2

    def plot_cell_evaluation(self, circle_insc_center=None,
                             circle_insc_radius=None, circle_circ_poly=None,
                             rect_circ_poly=None, inscribed_rect_poly=None,
                             expand_inscribed_rectangle=None) -> None:

        # plotting
        fig, ax = plt.subplots()

        # plot polygon
        x, y = self.poly.exterior.xy
        ax.plot(x, y, color='blue', label='Polygon')

        # plot centroid
        ax.plot(self.cxy[0], self.cxy[1], 'ro', label='Centroid')

        # plot mic
        if circle_insc_center is not None:
            ax.plot(circle_insc_center[0], circle_insc_center[1], 'go', label='MIC Center')
            circle = plt.Circle(circle_insc_center, circle_insc_radius, color='green', fill=False, label='MIC Radius')
            ax.add_artist(circle)

        # circle_circ_poly
        if circle_circ_poly is not None:
            mbc_x, mbc_y = circle_circ_poly.exterior.xy
            ax.plot(mbc_x, mbc_y, color='orange', label='Minimum Bounding Circle')

        # plot mrr
        if rect_circ_poly is not None:
            mrr_x, mrr_y = rect_circ_poly.exterior.xy
            ax.plot(mrr_x, mrr_y, color='purple', label='Minimum Rotated Rectangle')

        # inscribed_rect_poly
        if inscribed_rect_poly is not None:
            inscribed_rect_x, inscribed_rect_y = inscribed_rect_poly.exterior.xy
            ax.plot(inscribed_rect_x, inscribed_rect_y, color='cyan', label='Inscribed Rectangle')

        # expand_inscribed_rectangle
        if expand_inscribed_rectangle is not None:
            expand_rect_x, expand_rect_y = expand_inscribed_rectangle.exterior.xy
            ax.plot(expand_rect_x, expand_rect_y, color='magenta', label='Expanded Inscribed Rectangle')

        #
        ax.set_aspect('equal', adjustable='box')
        plt.legend()
        plt.title(f"Cell {self.id} Evaluation")
        plt.show()

    # #
    # if self.id == 11:
    #     fig, ax = plt.subplots()
    #     # main polygon
    #     x, y = self.poly.exterior.xy
    #     ax.plot(x, y, color='blue', label='Polygon')
    #     # rect circumscribed polygon
    #     x, y = inscribed_rect_poly.exterior.xy
    #     ax.plot(x, y, color='purple', label='inscribed_rect_poly')
    #     # scaled rectangle
    #     x, y = s_rect_poly.exterior.xy
    #     ax.plot(x, y, color='red', linestyle='--', label='inscribed_rect_poly')
    #     #
    #     x, y = expanded_rect_poly.exterior.xy
    #     ax.plot(x, y, color='magenta', label='Expanded Rectangle')
    #     ax.set_aspect('equal', adjustable='box')
    #     #
    #     plt.legend()
    #     plt.show()

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
