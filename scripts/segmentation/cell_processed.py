import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
from shapely.geometry import Point
from shapely.affinity import scale
from shapely import maximum_inscribed_circle
from scripts.base.logger import MyLogger
from scripts.segmentation.cell_base import CellBase
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import DistrictConfigs, PlanningConfigs
# from shapely import maximum_inscribed_circle, minimum_bounding_circle


class PCell(CellBase):
    """Processed cell class for segmentation and coverage path planning.
    add features for rectangle-like and inscribed rectangle-like evaluation.
    """

    def __init__(self, cell: CellBase, zmd: ZoneMapData) -> None:
        super().__init__(cell.id, cell.poly)
        self.lg = MyLogger(f"PCell-{self.id}")
        #
        self.rect_like: bool = False              # whether the cell is rectangle-like or inscribed rectangle-like
        self.rect_circ_like: bool = False         # whether the cell is rectangle-like based on circumscribed rectangle.
        self.rect_insc_like: bool = False         # whether the cell is inscribed rectangle-like or not
        self.rectangle_poly: Polygon = None       # inscribed rectangle polygon (expanded)

        # # major direction defined in base cell
        # self.major_direction: float = 0.0

        #
        self.zmd: ZoneMapData = zmd

        # settings
        self.robot_r_px = int(PlanningConfigs.cleaning_r_ov / zmd.resolution)
        self.dcfgs = DistrictConfigs()

        # find rectangle-like properties
        if self.dcfgs.use_insc_ret:
            self.eval_cell_rectangles()

        # find main directions
        self.find_cell_main_direction()

    def eval_cell_rectangles(self) -> None:
        """ Evaluate the cell for rectangle-like and inscribed rectangle-like properties """

        # circ: circumscribed
        # insc: inscribed

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

        # # eval rectangle-like
        self.eval_rectangle_like_circumscribed(rect_circ_area)  # rect_circ_like
        inscribed_rect_poly = self.create_inscribed_rectangle(rect_circ_poly, circle_insc_center, circle_insc_radius)
        self.rectangle_poly = self.expand_inscribed_rectangle(inscribed_rect_poly, circle_insc_center)
        self.rect_like = self.rect_insc_like

        # #
        # rect_area = self.rectangle_poly.area
        # rect_area_ratio = rect_area/self.area
        # print(f"rect_area_ratio: {rect_area_ratio:.2f}")
        # print(f"rect_like: {self.rect_like}")
        # # print(" --------------------- ")

        # #
        # self.plot_cell_evaluation(
        #     rect_circ_poly=rect_circ_poly,
        #     inscribed_rect_poly=inscribed_rect_poly,
        #     expand_inscribed_rectangle=self.rectangle_poly,
        #     circle_insc_center=circle_insc_center,
        #     circle_insc_radius=circle_insc_radius,
        #     # circle_circ_poly=circle_circ_poly,
        # )

    def find_cell_main_direction(self) -> None:
        """ Find main directions of each cell using Hough Transform.
            to be implemented in child classes.
        """
        if self.rect_like:
            self.major_direction = self.main_direction_by_cell_rectangle()
        else:
            map_cell = self.create_cell_map()
            self.major_direction = self.cell_main_direction(map_cell)

    # ==================================================================================================================
    # main direction of the cell
    # ==================================================================================================================

    def main_direction_by_cell_rectangle(self) -> float:
        """ Get main direction from cell rectangle orientation. """

        # Extract coordinates (last point == first point, so we skip last)
        rect = self.rectangle_poly
        coords = list(rect.exterior.coords)[:-1]

        longest_length = 0
        longest_angle = None

        for i, (x1, y1) in enumerate(coords):
            x2, y2 = coords[(i+1) % len(coords)]
            dx, dy = (x2 - x1, y2 - y1)
            length = np.hypot(dx, dy)

            if length > longest_length:
                longest_length = length
                # Angle in radians relative to +X axis
                longest_angle = np.arctan2(dy, dx)
        return (longest_angle + np.pi/2) % np.pi

    def create_cell_map(self) -> np.ndarray:
        """ Create binary map for the given cell. """

        # get polygon bounding box
        minx, miny, maxx, maxy = self.poly.bounds
        minx = minx - self.zmd.rectangle.x_min
        miny = miny - self.zmd.rectangle.y_min

        # crop map_ws to bounding box
        map_ws = self.zmd.map_ws.copy()
        map_ws = map_ws[int(miny):int(maxy), int(minx):int(maxx)]

        # create empty map
        map_cell = np.zeros_like(map_ws, dtype=np.uint8)

        # get polygon points of the cell
        points = self.points
        if self.dcfgs.use_insc_ret and self.rect_like:
            points = np.array(self.rectangle_poly.exterior.coords, dtype=np.int32)

        # fill cell polygon area with free space (255)
        # poly_pts = points - self.zmd.rectangle.min_point
        poly_pts = points - np.array([[minx, miny]], dtype=np.int32)
        cv.fillPoly(map_cell, [poly_pts], 255)

        # mask with free space in the original map
        map_cell = cv.bitwise_and(map_cell, map_ws)

        return map_cell

    def cell_main_direction(self, map_cell: np.ndarray) -> float:
        """ create MapPolygons object for the cell area in the map"""

        # Edge detection
        edges = cv.Canny(map_cell, 50, 150, apertureSize=3)

        # Hough Transform # tweak threshold if needed
        threshold = 120   # start threshold
        min_threshold = 50  # don't go below this
        lines = None
        while threshold >= min_threshold and lines is None:
            lines = cv.HoughLines(edges, 1, np.pi/180, threshold)
            if lines is None:
                threshold -= 20  # decrease threshold step-by-step

        # If no lines detected, fallback to default angle (e.g., 0)
        angles = []
        if lines is not None and len(lines) > 0:
            # rho, theta
            for _, theta in lines[:, 0]:
                angles.append(theta)
        else:
            # Fallback: assume horizontal direction
            angles.append(0.0)

        # Compute dominant wall orientation
        dominant_angle = np.median(angles)   # robust estimate
        # major_direction = (dominant_angle + np.pi/2) % np.pi
        major_direction = (dominant_angle) % np.pi

        # # Convert grayscale map to BGR for drawing
        # vis = cv.cvtColor(map_cell, cv.COLOR_GRAY2BGR)

        # # Draw arrow from center of map in free-space direction
        # h, w = map_cell.shape
        # center = (w//2, h//2)
        # length = min(w, h) // 3  # arrow length

        # dx = int(length * np.cos(major_direction))
        # dy = int(length * np.sin(major_direction))

        # cv.arrowedLine(vis, center, (center[0] + dx, center[1] + dy), (0, 0, 255), 3, tipLength=0.2)

        # # Show results
        # cv.imshow("Map with Direction", vis)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return major_direction

    # ==================================================================================================================
    # rectangle-like evaluation
    # ==================================================================================================================

    def eval_rectangle_like_circumscribed(self, rect_circ_area) -> None:
        """ Evaluate if the cell is rectangle-like based on the area ratio of the cell and the circumscribed rectangle. """
        k1 = 0.8
        cell_rect_circ_area_ratio = self.area/rect_circ_area
        self.rect_circ_like = cell_rect_circ_area_ratio >= k1

    def create_inscribed_rectangle(self, rect_circ_poly, circle_insc_center, circle_insc_radius) -> Polygon:
        """ Create an inscribed rectangle inside the circumscribed rectangle of the cell basde on the inscribed circle. """

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
        k1 = 0.74

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

    # ==================================================================================================================
    # rectangle expansion methods
    # ==================================================================================================================

    def expand_inscribed_rectangle_1(self, inscribed_rect_poly: Polygon, circle_insc_center) -> Polygon:
        """ expand in 4 directions. """

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
        """ expand based on the biggest edge. """

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
        l_edge = max_edge_idx / 4
        s_p1 = (p1[0] + l_edge * np.cos(th),
                p1[1] + l_edge * np.sin(th))
        s_p2 = (p2[0] + l_edge * np.cos(-th),
                p2[1] + l_edge * np.sin(-th))

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
        """ expand two points p1 and p2 in the direction of angle th by step and check if the new points are inside the polygon. """
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
        if self.poly.contains(Point(new_p1)):
            if self.poly.contains(Point(new_p2)):
                # point bertween p1 and p2
                mid_p = ((new_p1[0] + new_p2[0]) / 2,
                         (new_p1[1] + new_p2[1]) / 2)
                if self.poly.contains(Point(mid_p)):
                    success = True
        return success, new_p1, new_p2

    # ==================================================================================================================
    # plotting
    # ==================================================================================================================

    def plot_cell_evaluation(self, circle_insc_center=None,
                             circle_insc_radius=None, circle_circ_poly=None,
                             rect_circ_poly=None, inscribed_rect_poly=None,
                             expand_inscribed_rectangle=None) -> None:

        # plotting
        _, ax = plt.subplots()

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
