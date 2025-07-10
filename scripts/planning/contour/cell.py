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
    basic: bool                             # whether the cell is basic or not
    critical_points: list[CriticalPoint]    # critical points of the cell
    min_clearance: float                    # minimum clearance of the cell
    rect_circ_like: bool                    # whether the cell is rectangle-like or not
    rect_insc_like: bool                    # whether the cell is inscribed rectangle-like or not

    def __init__(self, ind: int, poly: Polygon, resolution: float = None) -> None:
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
        self.min_clearance = 0.0
        #
        self.rect_circ_like = False
        self.rect_insc_like = False

    def set_critical_points(self, critical_points: list[CriticalPoint]) -> None:
        """ set list of critical points in the cell and calculate minimum clearance """
        self.critical_points = critical_points
        self.find_min_clearance()

    def find_min_clearance(self) -> None:
        """ Find the minimum clearance of the cell based on critical points """
        if not self.critical_points:
            self.basic = True
            self.lg.warn("No critical points set for the cell, marking as basic.")
            return
        cp: CriticalPoint
        clearances = [cp.vp.clearance for cp in self.critical_points]
        self.min_clearance = min(clearances)

    # modify cell ==============================================================

    def process_cell(self) -> None:
        """ main function to process the cell """
        self.lg.info(f"Processing cell {self.id}")

        # modify the cell by removing points that are too close to critical points
        self.modify_cell()

        # # evaluate the cell for rectangle-like and inscribed rectangle-like properties
        # self.eval_cell()

    def modify_cell(self) -> None:
        """ Modify the cell by removing points that are too close to critical points """

        # # list of basis points from critical points with more than one pass (adjacent to other cells)
        # bsps = []  # basis points
        # cp: CriticalPoint
        # for cp in self.critical_points:
        #     if cp.n_passed > 1:
        #         bsps.append(cp.bs1_vp)
        # bsps.append(cp.bs2_vp)

        # remove poly points that are closer than the minimum clearance to the selected critical points (cps)
        min_d = self.robot_r_px * 0.6
        cell_points = self.poly.exterior.coords
        new_points = []
        for p in cell_points:
            p_valid = True
            for cp in self.critical_points:
                if cp.n_passed > 1:
                    dcp = np.linalg.norm(np.array(p) - np.array(cp.vp.v))
                    d1 = np.linalg.norm(np.array(p) - np.array(cp.bs1_vp))
                    d2 = np.linalg.norm(np.array(p) - np.array(cp.bs2_vp))
                    d = min(dcp, d1, d2)
                    if d < min_d:
                        p_valid = False
                        break
            if p_valid:
                new_points.append(p)
        #
        if len(new_points) > 3:
            self.poly = Polygon(new_points)
            self.create_cell_points()

    # eval cell ================================================================

    def eval_cell(self):
        """ Evaluate the cell for rectangle-like and inscribed rectangle-like properties """
        self.lg.info("Cell evaluation")

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
        self.eval_rectangle_like_circumscribed(rect_circ_area)
        inscribed_rect_poly = self.create_inscribed_rectangle(rect_circ_poly, circle_insc_center, circle_insc_radius)
        expanded_inscribed_rectangle = self.expand_inscribed_rectangle(inscribed_rect_poly, circle_insc_center)

        # print(f"cell/rect_circ area: {self.rect_circ_like}")
        # print(" --------------------- ")

        # #
        # self.plot_cell_evaluation(
        #     rect_circ_poly=rect_circ_poly,
        #     inscribed_rect_poly=inscribed_rect_poly,
        #     expand_inscribed_rectangle=expanded_inscribed_rectangle,
        #     circle_insc_center=circle_insc_center,
        #     circle_insc_radius=circle_insc_radius,
        #     # circle_circ_poly=circle_circ_poly,
        # )

    def eval_rectangle_like_circumscribed(self, rect_circ_area) -> None:
        """ Evaluate if the cell is rectangle-like based on the area ratio of the cell and the circumscribed rectangle """
        k1 = 0.8
        cell_rect_circ_area_ratio = self.area/rect_circ_area
        self.rect_circ_like = cell_rect_circ_area_ratio >= k1

    def create_inscribed_rectangle(self, rect_circ_poly, circle_insc_center, circle_insc_radius) -> Polygon:
        """ Create an inscribed rectangle inside the circumscribed rectangle of the cell basde on the inscribed circle """

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
        """ Expand the inscribed rectangle in 3 different ways and return the best one """
        k1 = 0.8

        # first expansion
        expanded_rect_poly_1 = self.expand_inscribed_rectangle_1(inscribed_rect_poly, circle_insc_center)
        area_1 = expanded_rect_poly_1.area
        rect_cell_a_r = area_1 / self.area
        # print(f"r1: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > k1:
            self.rect_insc_like = True
            return expanded_rect_poly_1

        # second expansion
        expanded_rect_poly_2 = self.expand_inscribed_rectangle_2(inscribed_rect_poly, circle_insc_center)
        area_2 = expanded_rect_poly_2.area
        rect_cell_a_r = area_2 / self.area
        # print(f"r2: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > k1:
            self.rect_insc_like = True
            return expanded_rect_poly_2

        # third expansion
        expanded_rect_poly_3 = self.expand_inscribed_rectangle_3(inscribed_rect_poly)
        area_3 = expanded_rect_poly_3.area
        rect_cell_a_r = area_3 / self.area
        # print(f"r3: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > k1:
            self.rect_insc_like = True
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
        n_e = np.ones(4, dtype=int)        # number of expansions in each direction

        # calculate the angles of the midpoints of the edges
        thetas = []
        for idx1, idx2 in zip(idx1s, idx2s):
            p1 = coords[idx1]
            p2 = coords[idx2]
            midp = np.mean([p1, p2], axis=0)
            th = np.arctan2(midp[1] - circle_insc_center[1], midp[0] - circle_insc_center[0])
            thetas.append(th)

        # expand in all directions
        for _ in range(2):
            new_coords = coords.copy()

            # update steps for each direction based on n_e
            max_n_e = np.max(n_e)
            if max_n_e == 0:
                break
            steps = [step*(ne/max_n_e) for step, ne in zip(steps, n_e)]

            any_valid = True
            while any_valid:
                any_valid = False
                for i, (idx1, idx2, th) in enumerate(zip(idx1s, idx2s, thetas)):
                    step = steps[i]
                    p1 = new_coords[idx1]
                    p2 = new_coords[idx2]
                    midp = np.mean([p1, p2], axis=0)
                    success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                    if success:
                        n_e[i] += 1
                        any_valid = True
                        new_coords[idx1] = new_p1
                        new_coords[idx2] = new_p2
                    elif steps[i] > min_step:
                        any_valid = True
                        steps[i] = step / 2  # reduce the step size

        # create the expanded rectangle polygon
        expanded_rect_poly = Polygon(new_coords)
        return expanded_rect_poly

    def expand_inscribed_rectangle_2(self, inscribed_rect_poly: Polygon, circle_insc_center) -> Polygon:
        """ first find the main direction of expansion buy expanding in each direction separately till collision, 
        then expand in that direction and then in the other 3 directions """

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
        n_e = np.zeros(4, dtype=int)  # number of expansions in each direction

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
                    n_e[i] += 1
                    coords_cand[i][idx1] = new_p1
                    coords_cand[i][idx2] = new_p2
                else:
                    is_valid = False

        # max step direction
        max_step_idx = np.argmax(n_e)
        best_coords = coords_cand[max_step_idx]

        # remove the max_step_idx item from idx1s, idx2s, thetas
        idx1s = [idx for i, idx in enumerate(idx1s) if i != max_step_idx]
        idx2s = [idx for i, idx in enumerate(idx2s) if i != max_step_idx]
        thetas = [th for i, th in enumerate(thetas) if i != max_step_idx]
        n_e = np.ones(3, dtype=int)        # reset

        # expand in the other 3 directions
        for _ in range(2):
            new_coords = best_coords.copy()  # reset best_coords for each iteration

            # update steps for each direction based on n_e
            max_n_e = np.max(n_e)
            if max_n_e == 0:
                break
            steps = [step*(ne/max_n_e) for step, ne in zip(steps, n_e)]

            any_valid = True
            while any_valid:
                any_valid = False
                for i, (idx1, idx2, th) in enumerate(zip(idx1s, idx2s, thetas)):
                    step = steps[i]
                    p1 = new_coords[idx1]
                    p2 = new_coords[idx2]
                    midp = np.mean([p1, p2], axis=0)
                    success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                    if success:
                        n_e[i] += 1
                        any_valid = True
                        new_coords[idx1] = new_p1
                        new_coords[idx2] = new_p2
                    elif steps[i] > min_step:
                        any_valid = True
                        steps[i] = step / 2  # reduce the step size

        # create the expanded rectangle polygon
        expanded_rect_poly = Polygon(new_coords)
        return expanded_rect_poly

    def expand_inscribed_rectangle_3(self, inscribed_rect_poly: Polygon) -> Polygon:
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
        th3 = th + np.pi / 2,
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
            self.lg.error("Cannot find a valid point p3 for the inscribed rectangle expansion.")
            return inscribed_rect_poly
            # raise ValueError("Cannot find a valid point p3 for the inscribed rectangle expansion.")

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
            self.lg.error("Cannot find a valid point p4 for the inscribed rectangle expansion.")
            return inscribed_rect_poly
            # raise ValueError("Cannot find a valid point p4 for the inscribed rectangle expansion.")

        # create new polygon
        coords = [p1, p2, p3, p4]  # new polygon
        new_poly = Polygon(coords)
        center = new_poly.centroid.coords[0]

        #
        coords = list(coords)
        idx1s = [1, 2, 3]
        idx2s = [2, 3, 0]
        steps = [step for _ in range(3)]
        n_e = np.ones(3, dtype=int)        # number of expansions in each direction

        # calculate the angles of the midpoints of the edges
        thetas = []
        for idx1, idx2 in zip(idx1s, idx2s):
            p1 = coords[idx1]
            p2 = coords[idx2]
            midp = np.mean([p1, p2], axis=0)
            th = np.arctan2(midp[1] - center[1], midp[0] - center[0])
            thetas.append(th)

        # expand in all directions
        for _ in range(2):
            new_coords = coords.copy()

            # update steps for each direction based on n_e
            max_n_e = np.max(n_e)
            if max_n_e == 0:
                break
            steps = [step*(ne/max_n_e) for step, ne in zip(steps, n_e)]

            #
            any_valid = True
            while any_valid:
                any_valid = False
                for i, (idx1, idx2, th) in enumerate(zip(idx1s, idx2s, thetas)):
                    step = steps[i]
                    p1 = new_coords[idx1]
                    p2 = new_coords[idx2]
                    midp = np.mean([p1, p2], axis=0)
                    success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, th, step)
                    if success:
                        n_e[i] += 1
                        any_valid = True
                        new_coords[idx1] = new_p1
                        new_coords[idx2] = new_p2
                    elif steps[i] > min_step:
                        any_valid = True
                        steps[i] = step / 2  # reduce the step size

        # create the expanded rectangle polygon
        expanded_rect_poly = Polygon(new_coords)
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
