""" Module for evaluating rectangle-like properties of polygons for planning optimization."""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
from shapely.geometry import Point
from shapely.affinity import scale
from shapely import maximum_inscribed_circle
from covpp.base import planning_utils as pu
from covpp.base.geometry import find_longest_edge_poly, find_shortest_edge_poly


class EvalRectangles:
    """Evaluate cell for rectangle-like properties to optimize planning.

    Pipeline Steps:
        1. Evaluate based on circumscribed rectangle (minimum_rotated_rectangle)
        2. Create base inscribed rectangle from maximum inscribed circle
        3. Expand inscribed rectangle in multiple directions
        4. Set rectangle-like property based on evaluation results
    """

    def __init__(self, cell_id: int, poly: Polygon, r_px: int) -> None:

        # args
        self.id = cell_id
        self.poly = poly
        self.spoly = deepcopy(poly)
        self.area = poly.area
        self.r_px = r_px

        # settings
        self.k_circum = 0.8
        self.k_inscr = 0.74

        # attributes for evaluation
        self.steps = {}                             # dictionary to hold expansion steps

        # rectangle-like properties
        self.rect_like: bool = False                # whether the cell is rectangle-like or inscribed rectangle-like
        self.rect_circum_like: bool = False         # whether the cell is rectangle-like based on circumscribed rectangle.
        self.rect_insc_like: bool = False           # whether the cell is inscribed rectangle-like or not
        # inscribed rectangle polygon (expanded)
        # also it is rotated to start from closest point to the start of the original polygon
        self.rectangle_poly: Polygon = None
        self.circum_rectangle_poly: Polygon = None  # circumscribed rectangle polygon

        # evaluate cell rectangles
        self.eval_cell_rectangles()

    def eval_cell_rectangles(self) -> None:
        """ Evaluate the cell for rectangle-like properties."""

        # circum: circumscribed
        # insc: inscribed

        # evaluate rectangle-like properties based on circumscribed rectangle (minimum_rotated_rectangle)
        circum_rect_poly = self.poly.minimum_rotated_rectangle
        self.circum_rectangle_poly = circum_rect_poly
        self.eval_by_circumscribed_rect()  # rect_circum_like

        # maximum_inscribed_circle
        insc_circle_poly = maximum_inscribed_circle(self.poly)
        insc_circle_center, circle_insc_boundary = insc_circle_poly.coords
        insc_circle_radius = np.linalg.norm(np.array(insc_circle_center) - np.array(circle_insc_boundary))

        # # minimum circumscribed circle
        # circle_circum_poly = minimum_bounding_circle(self.poly)
        # circle_circum_area = circle_circum_poly.area

        # create base inscribed rectangle
        base_inscribed_rect_poly = self.create_base_inscribed_rectangle(circum_rect_poly, insc_circle_center, insc_circle_radius)

        # calculate steps for expansion
        self.find_steps(base_inscribed_rect_poly, circum_rect_poly)

        # ----------------------------------------------------------------------
        # # eval rectangle-like based on
        rectangle_poly = self.eval_by_expand_inscribed_rectangle(base_inscribed_rect_poly, insc_circle_center)
        self.rect_like = self.rect_insc_like
        # ----------------------------------------------------------------------

        # start rectangle from closest point to the start of the original polygon
        # coords[:-1] so that last point (same as first) is not repeated
        points = np.array(self.poly.exterior.coords[:-1], dtype=np.int32)
        new_points = pu.rearrange_points_by_point(points[0], rectangle_poly.exterior.coords[:-1])
        self.rectangle_poly = Polygon(new_points)

        # # for debugging
        # rect_area = self.rectangle_poly.area
        # rect_area_ratio = rect_area/self.area
        # print(f"rect_area_ratio: {rect_area_ratio:.2f}")
        # print(f"rect_like: {self.rect_like}")
        # # print(" --------------------- ")

        # #
        # self.plot_cell_evaluation(
        #     circum_rect_poly=circum_rect_poly,
        #     inscribed_rect_poly=base_inscribed_rect_poly,
        #     expand_inscribed_rectangle=self.rectangle_poly,
        #     insc_circle_center=insc_circle_center,
        #     insc_circle_radius=insc_circle_radius,
        #     # circle_circum_poly=circle_circum_poly,
        # )

    def simplify_polygon(self, tol) -> Polygon:
        """ Simplify the cell polygon to reduce number of vertices. """
        simple_poly = self.poly.simplify(tol, preserve_topology=True)
        return simple_poly

    def find_steps(self, base_inscribed_rect_poly: Polygon, circum_rect_poly: Polygon) -> None:
        """ Find expansion steps based on cell properties. """
        steps = {}
        steps['robot_r'] = self.r_px
        steps['base_insc_rect'] = find_shortest_edge_poly(base_inscribed_rect_poly)['length'] / 10
        steps['circum_rect'] = find_shortest_edge_poly(circum_rect_poly)['length'] / 10
        self.steps = steps

    def eval_by_circumscribed_rect(self) -> None:
        """ Evaluate if the cell is rectangle-like based on the area ratio of the cell and the circumscribed rectangle. """
        cell_rect_circum_area_ratio = self.area/self.circum_rectangle_poly.area
        self.rect_circum_like = cell_rect_circum_area_ratio >= self.k_circum

    def create_base_inscribed_rectangle(self, circum_rect_poly, insc_circle_center, insc_circle_radius) -> Polygon:
        """ Create an inscribed rectangle inside the circumscribed rectangle of the cell based on the inscribed circle. """

        # calculate the angle of the rectangle
        rect_circum_coords = np.array(circum_rect_poly.exterior.coords)
        angle = np.arctan2(rect_circum_coords[1, 1] - rect_circum_coords[0, 1],
                           rect_circum_coords[1, 0] - rect_circum_coords[0, 0])

        # create initial rectangle inside the inscribed circle
        rect_coords = []
        theta = angle + np.pi / 4  # 45 degrees rotation for the rectangle
        for i in range(4):
            angle_offset = theta + i * np.pi / 2
            x = insc_circle_center[0] + insc_circle_radius * np.cos(angle_offset)
            y = insc_circle_center[1] + insc_circle_radius * np.sin(angle_offset)
            rect_coords.append((x, y))
        # create the rectangle polygon
        inscribed_rect_poly = Polygon(rect_coords)
        return inscribed_rect_poly

    def eval_by_expand_inscribed_rectangle(self, base_inscribed_rect_poly: Polygon, insc_circle_center) -> Polygon:
        """
        Evaluate the inscribed rectangle by comparing it three expanded inscribed rectangles.
        """

        # first expansion
        expanded_rect_poly_1 = self.expand_inscribed_rectangle_1(base_inscribed_rect_poly, insc_circle_center)
        area_1 = expanded_rect_poly_1.area
        rect_cell_a_r = area_1 / self.area
        # print(f"r1: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > self.k_inscr:
            self.rect_insc_like = True
            return expanded_rect_poly_1

        # second expansion
        expanded_rect_poly_2 = self.expand_inscribed_rectangle_2(base_inscribed_rect_poly, insc_circle_center)
        area_2 = expanded_rect_poly_2.area
        rect_cell_a_r = area_2 / self.area
        # print(f"r2: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > self.k_inscr:
            self.rect_insc_like = True
            return expanded_rect_poly_2

        # third expansion
        expanded_rect_poly_3 = self.expand_inscribed_rectangle_3(base_inscribed_rect_poly)
        area_3 = expanded_rect_poly_3.area
        rect_cell_a_r = area_3 / self.area
        # print(f"r3: {rect_cell_a_r:.2f}")
        if rect_cell_a_r > self.k_inscr:
            self.rect_insc_like = True
            return expanded_rect_poly_3

        #
        all_polys = [expanded_rect_poly_1, expanded_rect_poly_2, expanded_rect_poly_3]
        areas = [area_1, area_2, area_3]
        best_poly = max(zip(all_polys, areas), key=lambda x: x[1])[0]
        return best_poly

    # ==================================================================================================================
    # rectangle expansion methods
    # ==================================================================================================================

    def expand_inscribed_rectangle_1(self, inscribed_rect_poly: Polygon, insc_circle_center) -> Polygon:
        """ Expand the base inscribed rectangle.
            Keep expanding in each direction till collision, then move to next direction.
        """

        # find step
        # step = self.r_px
        step = self.steps['base_insc_rect']
        min_step = step / 5

        # scale inscribed_rect_poly
        sf = 0.8
        inscribed_rect_poly = scale(inscribed_rect_poly, xfact=sf, yfact=sf, origin=insc_circle_center)

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
            theta = np.arctan2(midp[1] - insc_circle_center[1], midp[0] - insc_circle_center[0])
            thetas.append(theta)

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
                for i, (idx1, idx2, theta) in enumerate(zip(idx1s, idx2s, thetas)):
                    step = steps[i]
                    p1 = new_coords[idx1]
                    p2 = new_coords[idx2]
                    midp = np.mean([p1, p2], axis=0)
                    success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, theta, step)
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

    def expand_inscribed_rectangle_2(self, inscribed_rect_poly: Polygon, insc_circle_center) -> Polygon:
        """
        Expand the base inscribed rectangle.
        First find the main direction of expansion buy expanding in each direction separately till collision,
        then expand in that direction and then in the other directions.
        """

        # find step
        # smallest length of the rectangle side
        rect_circum_coords = np.array(inscribed_rect_poly.exterior.coords)
        rect_circum_side_lengths = np.linalg.norm(np.diff(rect_circum_coords, axis=0), axis=1)
        min_rect_circum_side_length = np.min(rect_circum_side_lengths)
        step = self.steps['base_insc_rect']
        min_step = step / 5

        # scale inscribed_rect_poly
        sf = 0.3
        s_rect_poly = scale(inscribed_rect_poly, xfact=sf, yfact=sf, origin=insc_circle_center)

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
            theta = np.arctan2(midp[1] - insc_circle_center[1], midp[0] - insc_circle_center[0])
            thetas.append(theta)

        # find main direction of expansion
        # expand seperately in each direction till collision
        coords_cand = [coords.copy() for _ in range(4)]
        for i, (idx1, idx2, theta) in enumerate(zip(idx1s, idx2s, thetas)):
            is_valid = True
            while is_valid:
                step = steps[i]
                p1 = coords_cand[i][idx1]
                p2 = coords_cand[i][idx2]
                midp = np.mean([p1, p2], axis=0)
                success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, theta, step)
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
        thetas = [theta for i, theta in enumerate(thetas) if i != max_step_idx]
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
                for i, (idx1, idx2, theta) in enumerate(zip(idx1s, idx2s, thetas)):
                    step = steps[i]
                    p1 = new_coords[idx1]
                    p2 = new_coords[idx2]
                    midp = np.mean([p1, p2], axis=0)
                    success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, theta, step)
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
        """
        Expand the base inscribed rectangle.
        Expand based on the biggest edge of the base rectangle.
        """

        # step
        # step = self.r_px
        step = self.steps['base_insc_rect']
        min_step = step / 3

        # find biggest edge of the self.poly
        edge_prop = find_longest_edge_poly(self.poly)
        p1 = edge_prop['p1']
        p2 = edge_prop['p2']
        theta = edge_prop['angle']
        edge_length = edge_prop['length']

        #  move p1 and p2 in the direction of the center of the edge
        l_edge = edge_length / 4
        s_p1 = (p1[0] + l_edge * np.cos(theta),
                p1[1] + l_edge * np.sin(theta))
        s_p2 = (p2[0] + l_edge * np.cos(-theta),
                p2[1] + l_edge * np.sin(-theta))

        # create third and fourth points
        # p3 is the point from p2 perpendicular to edge p1-p2
        th3 = theta + np.pi / 2,
        p3 = (
            s_p2[0] + step * np.cos(th3),
            s_p2[1] + step * np.sin(th3)
        )
        if not self.spoly.contains(Point(p3)):
            th3 = theta - np.pi / 2
            p3 = (
                s_p1[0] + step * np.cos(th3),
                s_p1[1] + step * np.sin(th3)
            )
        if not self.spoly.contains(Point(p3)):
            # self.lg.error("[expand_inscribed_rectangle_3] Cannot find a valid point p3.")
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
        if not self.spoly.contains(Point(p4)):
            # self.lg.error("Cannot find a valid point p4 for the inscribed rectangle expansion.")
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
            theta = np.arctan2(midp[1] - center[1], midp[0] - center[0])
            thetas.append(theta)

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
                for i, (idx1, idx2, theta) in enumerate(zip(idx1s, idx2s, thetas)):
                    step = steps[i]
                    p1 = new_coords[idx1]
                    p2 = new_coords[idx2]
                    midp = np.mean([p1, p2], axis=0)
                    success, new_p1, new_p2 = self.expand_2_points_eval(p1, p2, theta, step)
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

    def expand_2_points_eval(self, p1, p2, theta, step):
        """ expand two points p1 and p2 in the direction of angle theta by step and check if the new points are inside the polygon. """
        new_p1 = (
            p1[0] + step * np.cos(theta),
            p1[1] + step * np.sin(theta)
        )
        new_p2 = (
            p2[0] + step * np.cos(theta),
            p2[1] + step * np.sin(theta)
        )
        # check if the new points are inside the polygon
        success = False
        if self.spoly.contains(Point(new_p1)):
            if self.spoly.contains(Point(new_p2)):
                # point bertween p1 and p2
                mid_p = ((new_p1[0] + new_p2[0]) / 2,
                         (new_p1[1] + new_p2[1]) / 2)
                if self.spoly.contains(Point(mid_p)):
                    success = True
        return success, new_p1, new_p2

    # ==================================================================================================================
    # plotting
    # ==================================================================================================================

    def plot_cell_evaluation(self, insc_circle_center=None,
                             insc_circle_radius=None, circle_circum_poly=None,
                             circum_rect_poly=None, inscribed_rect_poly=None,
                             expand_inscribed_rectangle=None) -> None:
        """ Plot the rectangle evaluation results. """

        # plotting
        _, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 8))
        # dont show axis
        ax.axis('off')
        # fix aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # plot polygon
        x, y = self.spoly.exterior.xy
        ax.plot(x, y, color='blue', label='Cell Polygon', linewidth=2, linestyle='-')

        # plot centroid
        cxy = self.poly.centroid.coords[0]
        ax.plot(cxy[0], cxy[1], 'ro', label='Polygon Centroid')

        # plot mic
        if insc_circle_center is not None:
            ax.plot(insc_circle_center[0], insc_circle_center[1], 'go', label='MIC Center')
            circle = plt.Circle(
                insc_circle_center,
                insc_circle_radius,
                color='green',
                fill=False,
                label='MIC Circle',
                linestyle=':',
                linewidth=2
            )
            ax.add_artist(circle)

        # circle_circum_poly
        if circle_circum_poly is not None:
            mbc_x, mbc_y = circle_circum_poly.exterior.xy
            ax.plot(mbc_x, mbc_y, color='orange', label='MIC', linestyle=(0, (3, 5, 1, 5)))

        # plot mrr
        if circum_rect_poly is not None:
            mrr_x, mrr_y = circum_rect_poly.exterior.xy
            ax.plot(mrr_x, mrr_y, color='purple', label='MRB', linestyle='-.', linewidth=2)

        # inscribed_rect_poly
        if inscribed_rect_poly is not None:
            inscribed_rect_x, inscribed_rect_y = inscribed_rect_poly.exterior.xy
            ax.plot(inscribed_rect_x, inscribed_rect_y, color='cyan', label=' Initial MIR', linestyle='--', linewidth=2)

        # expand_inscribed_rectangle
        if expand_inscribed_rectangle is not None:
            expand_rect_x, expand_rect_y = expand_inscribed_rectangle.exterior.xy
            ax.plot(expand_rect_x, expand_rect_y, color='magenta', label='Estimated MIR', linewidth=2, linestyle=(0, (5, 1)))

        ax.invert_yaxis()

        # legend at outer upper right (outside the plot)
        plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.9), fontsize=12, frameon=True)
        # plt.show()

        # save_dir = "./maps/zOther/segmentation"
        # fig.savefig(f"{save_dir}/cell_{self.id}_evaluation.pdf", bbox_inches='tight', format='pdf', dpi=600)
        # fig.savefig(f"{save_dir}/cell_{self.id}_evaluation.pdf", format='pdf', dpi=600)
