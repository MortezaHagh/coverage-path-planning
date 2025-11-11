"""Contour Line-based Coverage Path Planning implementation."""
import cv2 as cv
import numpy as np
# from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from covpp.base.logger import MyLogger
from covpp.base.base_utils import MyPoint, MyPath
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.voronoi_segmentation.cell import Cell
from covpp.planning.coverage_pp_base import CoveragePathPlannerBase


class CoveragePathPlanner(CoveragePathPlannerBase):
    """Coverage path planner using contour-following approach.

    This planner generates spiral or contour-following paths by progressively
    shrinking the cell polygon or using distance transforms.

    Planning Methods:
        - Polygon buffering: Iterative inward scaling
        - Distance transform: Skeleton-based path generation
        - Combination: Hybrid approach for complex shapes

    Result:
        Sets path_m and path_px attributes for the cell.
    """

    def __init__(self, cell: Cell, zmd: ZoneMapData):
        super().__init__(cell, zmd)
        self.lg = MyLogger(f"CellCoveragePlannerContour-Z{zmd.ind}-C{cell.id}")
        self.coverage_planning()

    def coverage_planning(self) -> None:
        """Execute contour-following coverage planning for the cell."""
        if not self.cell.do_planning:
            self.lg.warn(f"cell {self.cell.id} does not require planning.")
            return
        # self.calculate_path_by_poly_buffer()
        self.calculate_path_combinaiton()
        self.create_path_m()

    def calculate_path_by_poly_buffer(self) -> None:
        """Calculate coverage path by iteratively shrinking the cell polygon.

        Uses polygon buffering to create concentric contours that form a
        spiral coverage pattern.
        """
        poly = self.cell.poly
        # scale the polygon to smaller size

        robot_r_px = self.pcfg.cleaning_r_ov / self.resolution
        robot_d_px = robot_r_px * 2

        scaled_polys = []
        points = []

        # scaled_poly = scale(poly, xfact=0.5, yfact=0.5, origin='center')

        #
        scaled_poly = poly.buffer(-1, join_style=2, resolution=8)  # initial scale

        #
        i = 1
        while not (scaled_poly.is_empty or scaled_poly is None):
            #  if result is MultiPolygon
            if isinstance(scaled_poly, MultiPolygon):
                scaled_poly = scaled_poly.geoms[0]  # take the first polygon
            scaled_polys.append(scaled_poly)
            new_points = scaled_poly.exterior.coords
            points.extend(list(new_points))
            scaled_poly = poly.buffer(-robot_d_px*i, join_style=2, resolution=16)  # scale down the polygon
            i += 1

        # create a path from the scaled polygon
        points = [MyPoint(int(p[0]), int(p[1])) for p in points]
        self.path_px = MyPath(points)

    def calculate_path_combinaiton(self) -> None:
        """ Calculate the coverage path by distance transform and inscribed rectangle if available. """

        robot_r_px = int(self.pcfg.cleaning_r_ov / self.resolution)
        points = []

        # plan by inscribed rectangle if available
        if self.cell.rect_like:
            self.path_px = self.complete_path_by_rectangle(points, robot_r_px)
            return

        # initial scale of the polygon
        scaled_poly = self.cell.poly.buffer(-1, join_style=2, resolution=8)
        if scaled_poly.is_empty or scaled_poly is None:
            raise ValueError(f"Cell {self.cell.id} has an empty scaled_poly.")
        if isinstance(scaled_poly, MultiPolygon):
            # take the first polygon if multiple polygons exist
            scaled_poly = scaled_poly.geoms[0]
        points.extend(list(scaled_poly.exterior.coords))

        # shift in the rectangle coordinates
        s_xy = self.cell.global_xy

        # complete the path by distance transform
        path_px = self.complete_path_by_distance_transform(points, self.cell.map, s_xy, robot_r_px)
        self.path_px = path_px

    def complete_path_by_rectangle(self, points: list, robot_r_px: int) -> MyPath:
        """
        Complete the path using the inscribed rectangle of the cell.

        Steps:
        1. Start from the inscribed rectangle and scale it down iteratively.
        2. Rearrange and add rectangle points to the path.
        3. Handle the center line if necessary (e.g., for odd-width rectangles).
        4. Create the final path.
        """

        rect = self.cell.rectangle_poly
        robot_d_px = int(robot_r_px * 2)

        # start rectangle from closest point to the start of the original polygon
        poly_points = self.cell.poly.exterior.coords
        rec_points = rect.exterior.coords
        new_points = self.rearrange_points(poly_points[0], rec_points)
        rect = Polygon(new_points)

        # main rectangle
        scaled_rect = rect.buffer(-4, join_style=2, resolution=8)
        if scaled_rect.is_empty or scaled_rect is None:
            self.lg.error(f"Cell {self.cell.id} main_rectangle has issues.")
            return MyPath([])
        new_points = scaled_rect.exterior.coords
        points.extend(list(new_points))

        # n scales
        # find length of the smaller edge of the rectangle
        coords = np.array(scaled_rect.exterior.coords)
        edges = np.diff(coords, axis=0)
        edge_lengths = np.linalg.norm(edges, axis=1)
        min_edge = min(edge_lengths)

        # find number of scales
        # case1
        n1 = ((min_edge / robot_d_px) - 1.0) / 2
        dd = n1 - n1 // 1
        if dd > 0.1:
            n1 = np.ceil(n1)
        else:
            n1 = int(n1)
        new_d_1 = min_edge / (2*n1 + 1.0)
        # case2
        n2 = (min_edge / (2*robot_d_px)) - 1.0
        dd = n2 - n2 // 1
        if dd > 0.1:
            n2 = np.ceil(n2)
        else:
            n2 = int(n2)
        new_d_2 = min_edge / (2*n2 + 2.0)

        # choose the smaller number of scales
        case = 0
        if new_d_1 > new_d_2:
            case = 1
            n = n1
            new_d = new_d_1
        else:
            case = 2
            n = n2
            new_d = new_d_2

        # scaling
        for _ in range(int(n)):
            scaled_rect = scaled_rect.buffer(-new_d, join_style=2, resolution=8)
            if scaled_rect.is_empty or scaled_rect is None:
                break
            new_points = scaled_rect.exterior.coords
            new_points = self.rearrange_points(points[-1], new_points)
            points.extend(list(new_points))

        # line like at the center of the rectangle
        if case == 2:
            last_scale_c = min_edge/2 - 1
            scaled_rect = rect.buffer(-last_scale_c, join_style=2, resolution=8)
            if not (scaled_rect.is_empty or scaled_rect is None):
                new_points = scaled_rect.exterior.coords
                new_points = self.rearrange_points(points[-1], new_points)
                # remove next big edge
                edges = np.diff(new_points, axis=0)
                edge_lengths = np.linalg.norm(edges, axis=1)
                idx = 2
                if edge_lengths[1] > edge_lengths[0]:
                    idx = 3
                points.extend(list(new_points)[:idx])

        # final path
        points = [MyPoint(int(p[0]), int(p[1])) for p in points]
        return MyPath(points)

    def complete_path_by_distance_transform(self, points: list, c_map: np.ndarray, s_xy: np.ndarray, robot_r_px: int) -> MyPath:
        """
        Complete the path using distance transform method.

        Steps:
        1. Compute the distance transform of the cell map.
        2. Iteratively find contours at increasing distances from the boundary.
        3. Rearrange and add contour points to the path.
        4. Handle any remaining area with a final contour extraction.
        """

        robot_d_px = int(robot_r_px * 2)

        # distance transform
        dist_transform = cv.distanceTransform(c_map, cv.DIST_L2, 5)

        # max value in the distance transform
        curr_dist = 0
        max_dist = dist_transform.max()

        # calculate the number of lines ****
        n = (max_dist / robot_d_px) - 0.5
        dd = n - n//1
        if dd > 0.1:
            n = np.ceil(n)
        else:
            n = int(n)
            n = int(n)
        new_d = max_dist / (n+0.5)

        # find polygon for each distance
        while curr_dist < (max_dist-new_d):
            curr_dist += new_d

            polys = self.find_poly_by_dist(dist_transform=dist_transform, dist=curr_dist)
            if polys is None or len(polys) == 0:
                break

            # rearrange polygons by area
            polys = self.rearrange_polys_by_area(polys)

            # add points from the polygons to the path
            for poly in polys:
                new_points = poly.exterior.coords + s_xy
                # rearrange points starting from the closest point to the last point in points
                new_points = self.rearrange_points(points[-1], new_points)
                points.extend(list(new_points))

        # last try
        if (max_dist - curr_dist) > robot_r_px:
            r_area = robot_r_px * robot_r_px * np.pi / 2
            polys = self.find_poly_by_dist(dist_transform=dist_transform, dist=max_dist)
            if polys is not None and len(polys) > 0:
                for poly in polys:
                    coords = poly.exterior.coords
                    if poly.area < r_area:
                        n_coords = int(len(coords) / 2)
                        coords = np.array(coords[:n_coords], dtype=np.float32)

                    new_points = coords + s_xy
                    # rearrange points starting from the closest point to the last point in points
                    new_points = self.rearrange_points(points[-1], new_points)
                    points.extend(list(new_points))

        # create a path from the scaled polygon
        points = [MyPoint(int(p[0]), int(p[1])) for p in points]
        return MyPath(points)

    def find_poly_by_dist(self, dist_transform: np.ndarray, dist: int, tolerance: int = 1) -> list[Polygon]:
        """ Find polygons in the distance transform map that are close to the target distance. """

        # Binary mask of pixels close to target distance
        mask = (dist_transform >= dist - tolerance) & (dist_transform <= dist + tolerance)

        # Convert boolean mask to uint8
        mask = mask.astype(np.uint8) * 255

        # Optional: clean up mask (remove noise)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.lg.error(f"No contours found for cell {self.cell.id}.")
            return
        #
        polys = []
        for cnt in contours:
            if len(cnt) < 3:
                self.lg.warn("Contour has less than 3 points, skipping.")
                continue
            # create a polygon from the contour points
            dist_points = [tuple(p[0]) for p in cnt]
            poly = Polygon(dist_points)
            polys.append(poly)
        return polys

    # ==========================================================================

    def rearrange_polys_by_area(self, polys: list[Polygon]) -> list[Polygon]:
        """ Rearrange polygons by area in descending order. """
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys

    def rearrange_points(self, point: np.ndarray, points: np.ndarray) -> np.ndarray:
        """ Rearrange points starting from the closest point to the given point. """
        # Find the index of the closest point to the given point
        idx = self.find_closest_point_idx(point, points)
        # Rearrange points starting from the closest point
        rearranged_points = self.rearrange_points_by_idx(idx, points)
        # add first point to the end of the list
        rearranged_points = np.concatenate((rearranged_points, [rearranged_points[0]]), axis=0)
        return rearranged_points

    def rearrange_points_by_idx(self, idx: int, points: np.ndarray) -> np.ndarray:
        """ Rearrange points starting from idx. """
        if idx < 0 or idx >= len(points):
            self.lg.error(f"Index {idx} is out of bounds for points list.")
            return points
        if idx == 0:
            return points
        return np.concatenate((points[idx:], points[:idx]), axis=0)

    def find_closest_point_idx(self, p: np.ndarray, points: np.ndarray) -> int:
        """ Find the index of the closest point to p in points. """
        dists = [np.linalg.norm(np.array(p) - np.array(pt)) for pt in points]
        return np.argmin(dists)
