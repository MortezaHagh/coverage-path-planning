""" Module for evaluating rectangle-like properties of polygons for planning optimization."""
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
from shapely.geometry import Point
from shapely.affinity import scale, rotate, translate
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

    def __init__(self, cell_id: int, poly: Polygon, radius_px: int) -> None:

        # args
        self.id = cell_id
        self.poly = poly
        self.spoly = deepcopy(poly)
        self.area = poly.area
        self.radius_px = radius_px

        # settings
        self.k_circum = 0.8
        self.k_inscr = 0.77  # 0.77 # [m1:0.7]

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

    def find_contact_angle(self, insc_circle_poly) -> list[float]:
        """ Find the angle of contact point between cell polygon and inscribed circle polygon. 

            Used to create the base inscribed rectangle aligned with the cell shape.
        """
        angles = []

        # find the contact points between two shapely polygons
        contact_points = self.poly.intersection(insc_circle_poly)
        if contact_points.is_empty:
            raise ValueError("No contact points found between cell polygon and inscribed circle polygon.")
        contact_coords = np.array(contact_points.coords)
        contact_point = contact_coords[0]  # take the first contact point
        #  find the angle of the contact point with respect to the center of the inscribed circle
        insc_circle_center = insc_circle_poly.centroid.coords[0]
        angle = np.arctan2(contact_point[1] - insc_circle_center[1],
                           contact_point[0] - insc_circle_center[0])
        angles.append(angle)

        # alternative: use the circumscribed rectangle to find the angle
        # calculate the angle of the rectangle
        rect_circum_coords = np.array(self.poly.minimum_rotated_rectangle.exterior.coords)
        angle = np.arctan2(rect_circum_coords[1, 1] - rect_circum_coords[0, 1],
                           rect_circum_coords[1, 0] - rect_circum_coords[0, 0])
        angles.append(angle)
        return angles

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

        # find angle for initial base_inscribed_rect_poly
        angles = self.find_contact_angle(insc_circle_poly)

        # # minimum circumscribed circle
        # circle_circum_poly = minimum_bounding_circle(self.poly)
        # circle_circum_area = circle_circum_poly.area

        insc_rect_polys = []
        angles.append(0)
        poly_center = self.poly.centroid.coords[0]
        centers = [insc_circle_center, insc_circle_center, poly_center]

        for angle_, center_ in zip(angles, centers):
            # create base inscribed rectangle
            base_inscribed_rect_poly = self.create_base_inscribed_rectangle(angle_, center_, insc_circle_radius * 0.8)
            # calculate steps for expansion
            self.find_steps(base_inscribed_rect_poly, circum_rect_poly)

            # ----------------------------------------------------------------------
            # # eval rectangle-like based on
            insc_rect_poly = self.eval_by_expand_inscribed_rectangle(base_inscribed_rect_poly, center_)
            insc_rect_polys.append(insc_rect_poly)
            # ----------------------------------------------------------------------

        # choose the best rectangle polygon based on area
        areas = [rect_poly.area for rect_poly in insc_rect_polys]
        max_area_idx = np.argmax(areas)
        rectangle_poly = insc_rect_polys[max_area_idx]
        self.rect_insc_like = areas[max_area_idx] / self.area >= self.k_inscr
        self.rect_like = self.rect_insc_like

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
        # print(" --------------------- ")

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
        steps['robot_r'] = self.radius_px
        steps['base_insc_rect'] = find_shortest_edge_poly(base_inscribed_rect_poly)['length'] / 8
        steps['circum_rect'] = find_shortest_edge_poly(circum_rect_poly)['length'] / 10
        steps['scale_factor'] = 0.08  # 0.08 # [m1:0.9]
        self.steps = steps

    def eval_by_circumscribed_rect(self) -> None:
        """ Evaluate if the cell is rectangle-like based on the area ratio of the cell and the circumscribed rectangle. """
        cell_rect_circum_area_ratio = self.area / self.circum_rectangle_poly.area
        self.rect_circum_like = cell_rect_circum_area_ratio >= self.k_circum

    def create_base_inscribed_rectangle(self, angle: float, insc_circle_center, insc_circle_radius) -> Polygon:
        """ Create an inscribed rectangle inside the circumscribed rectangle of the cell based on the inscribed circle. """

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
        # rect_cell_a_r = area_1 / self.area
        # print(f"r1: {rect_cell_a_r:.2f}")
        # if rect_cell_a_r > self.k_inscr:
        #     return expanded_rect_poly_1, True

        # second expansion
        expanded_rect_poly_2 = self.expand_inscribed_rectangle_2(base_inscribed_rect_poly, insc_circle_center)
        area_2 = expanded_rect_poly_2.area
        # rect_cell_a_r = area_2 / self.area
        # print(f"r2: {rect_cell_a_r:.2f}")
        # if rect_cell_a_r > self.k_inscr:
        #     return expanded_rect_poly_2, True

        # third expansion
        expanded_rect_poly_3 = self.expand_inscribed_rectangle_3(base_inscribed_rect_poly)
        area_3 = expanded_rect_poly_3.area
        # rect_cell_a_r = area_3 / self.area
        # print(f"r3: {rect_cell_a_r:.2f}")
        # if rect_cell_a_r > self.k_inscr:
        #     return expanded_rect_poly_3, True

        #
        all_polys = [expanded_rect_poly_1, expanded_rect_poly_2, expanded_rect_poly_3]
        areas = [area_1, area_2, area_3]
        best_poly = max(zip(all_polys, areas), key=lambda x: x[1])[0]
        return best_poly

    # ==================================================================================================================
    # rectangle expansion methods
    # ==================================================================================================================

    def expand_inscribed_rectangle_1(self, inscribed_rect_poly: Polygon, insc_circle_center) -> Polygon:
        """Expand by iteratively scaling along each edge direction using containment checks.
        Uses `scale_rect_poly` for directional scaling and `is_poly_in_polygon` to stop at collision.
        """

        # Initial gentle shrink to provide room for expansion stability
        inscribed_rect_poly = scale(inscribed_rect_poly, xfact=0.8, yfact=0.8, origin=insc_circle_center)

        # Determine small scaling step relative to rectangle size
        delta_scale = self.steps.get('scale_factor', 0.1)
        # shortest_len = find_shortest_edge_poly(inscribed_rect_poly)['length']
        # base_step = self.steps.get('base_insc_rect', shortest_len / 8.0)
        # delta_scale = max(0.01, min(0.08, base_step / max(shortest_len, 1e-6) / 2.0))

        # Calculate edge midpoints angles
        coords = list(inscribed_rect_poly.exterior.coords[:-1])
        idx1s = [0, 1, 2, 3]
        idx2s = [1, 2, 3, 0]
        thetas = []
        for idx1, idx2 in zip(idx1s, idx2s):
            p1 = coords[idx1]
            p2 = coords[idx2]
            midp = np.mean([p1, p2], axis=0)
            theta = np.arctan2(midp[1] - insc_circle_center[1], midp[0] - insc_circle_center[0])
            thetas.append(theta)

        # Two passes to allow iterative balancing of directions
        for _ in range(2):
            for theta in thetas:
                # Expand along current direction until collision
                local_delta = delta_scale
                grew = True
                while grew:
                    candidate = self.scale_rect_poly(inscribed_rect_poly, 1.0 + local_delta, theta)
                    if self.is_poly_in_polygon(candidate):
                        inscribed_rect_poly = candidate
                    else:
                        grew = False
                    #     # try smaller increment
                    #     local_delta *= 0.5
                    #     if local_delta < 0.005:
                    #         grew = False

        return inscribed_rect_poly

    def expand_inscribed_rectangle_2(self, inscribed_rect_poly: Polygon, insc_circle_center) -> Polygon:
        """Expand by selecting the most permissive direction first, then others.
        Uses `scale_rect_poly` for directional scaling and `is_poly_in_polygon` to bound growth.
        """

        # Gentle initial shrink for stability
        s_rect_poly = scale(inscribed_rect_poly, xfact=0.6, yfact=0.6, origin=insc_circle_center)

        # Determine scaling step
        delta_scale = self.steps.get('scale_factor', 0.1)
        # shortest_len = find_shortest_edge_poly(s_rect_poly)['length']
        # base_step = self.steps.get('base_insc_rect', shortest_len / 8.0)
        # delta_scale = max(0.01, min(0.08, base_step / max(shortest_len, 1e-6) / 2.0))

        # Edge midpoints angles
        coords = list(s_rect_poly.exterior.coords[:-1])
        idx1s = [0, 1, 2, 3]
        idx2s = [1, 2, 3, 0]
        thetas = []
        for idx1, idx2 in zip(idx1s, idx2s):
            p1 = coords[idx1]
            p2 = coords[idx2]
            midp = np.mean([p1, p2], axis=0)
            theta = np.arctan2(midp[1] - insc_circle_center[1], midp[0] - insc_circle_center[0])
            thetas.append(theta)

        # Probe each direction separately to measure allowable growth
        growth_counts = []
        for theta in thetas:
            test_poly = s_rect_poly
            count = 0
            local_delta = delta_scale
            while True:
                candidate = self.scale_rect_poly(test_poly, 1.0 + local_delta, theta)
                if self.is_poly_in_polygon(candidate):
                    test_poly = candidate
                    count += 1
                else:
                    break
            growth_counts.append(count)

        # Choose best direction and expand fully along it
        best_idx = int(np.argmax(growth_counts)) if len(growth_counts) > 0 else 0
        best_theta = thetas[best_idx]
        local_delta = delta_scale
        while True:
            candidate = self.scale_rect_poly(s_rect_poly, 1.0 + local_delta, best_theta)
            if self.is_poly_in_polygon(candidate):
                s_rect_poly = candidate
            else:
                break

        # Expand remaining directions modestly
        for theta in [t for i, t in enumerate(thetas) if i != best_idx]:
            local_delta = delta_scale
            grew = True
            while grew:
                candidate = self.scale_rect_poly(s_rect_poly, 1.0 + local_delta, theta)
                if self.is_poly_in_polygon(candidate):
                    s_rect_poly = candidate
                else:
                    grew = False
                #     local_delta *= 0.5
                #     if local_delta < 0.005:
                #         grew = False

        return s_rect_poly

    def expand_inscribed_rectangle_3(self, inscribed_rect_poly: Polygon) -> Polygon:
        """Expand primarily along the longest edge direction of the cell polygon.
        Uses `scale_rect_poly` with the edge's angle and its perpendicular.
        """

        # Determine scaling step
        delta_scale = self.steps.get('scale_factor', 0.1)
        # shortest_len = find_shortest_edge_poly(inscribed_rect_poly)['length']
        # base_step = self.steps.get('base_insc_rect', shortest_len / 8.0)
        # delta_scale = max(0.01, min(0.08, base_step / max(shortest_len, 1e-6) / 2.0))

        # Longest edge properties of the cell polygon
        edge_prop = find_longest_edge_poly(self.poly)
        theta = edge_prop['angle']

        # First, expand along longest-edge direction
        grew = True
        while grew:
            candidate = self.scale_rect_poly(inscribed_rect_poly, 1.0 + delta_scale, theta)
            if self.is_poly_in_polygon(candidate):
                inscribed_rect_poly = candidate
            else:
                grew = False
                # delta_scale *= 0.5
                # if delta_scale < 0.005:
                #     grew = False

        # Then, expand along the perpendicular direction
        perp_theta = theta + np.pi / 2.0
        local_delta = max(delta_scale, 0.01)
        grew = True
        while grew:
            candidate = self.scale_rect_poly(inscribed_rect_poly, 1.0 + local_delta, perp_theta)
            if self.is_poly_in_polygon(candidate):
                inscribed_rect_poly = candidate
            else:
                grew = False
            #     local_delta *= 0.5
            #     if local_delta < 0.005:
            #         grew = False

        return inscribed_rect_poly

    def is_poly_in_polygon(self, poly: Polygon) -> bool:
        """ Check if the polygon poly is inside the cell polygon. """
        for coord in poly.exterior.coords:
            point = Point(coord)
            if not self.spoly.contains(point):
                return False
        return True

    def scale_rect_poly(self, rect_poly: Polygon, scale_factor: float, theta: float) -> Polygon:
        """ Scale the rectangle polygon in the direction of theta by scale_factor. """
        # center of the rectangle
        center = rect_poly.centroid.coords[0]
        # rotate to align with x axis
        rect_poly = rotate(rect_poly, angle=np.degrees(-theta), origin=center)
        # scale in x direction
        scaled_rect_poly = scale(rect_poly, xfact=scale_factor, yfact=1.0, origin=center)
        # rotate back
        scaled_rect_poly = rotate(scaled_rect_poly, angle=np.degrees(theta), origin=center)
        return scaled_rect_poly

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
        plt.show()

        # save_dir = "./maps/zOther/segmentation"
        # fig.savefig(f"{save_dir}/cell_{self.id}_evaluation.pdf", bbox_inches='tight', format='pdf', dpi=600)
        # fig.savefig(f"{save_dir}/cell_{self.id}_evaluation.pdf", format='pdf', dpi=600)
