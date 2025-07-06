""" contour based coverage planning inside the given cell """
import cv2 as cv
import numpy as np
# from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon
from scripts.base.logger import MyLogger
from scripts.planning.base_cell import BaseCell
from scripts.base.configs import PlanningConfigs
from scripts.base.my_utils import MyPoint, MyPath
from scripts.district.zone_map_data import ZoneMapData


class CellCoveragePathPlanner:
    cell: BaseCell
    zmd: ZoneMapData
    pcfgs: PlanningConfigs
    resolution: float
    path_m: MyPath
    path_px: MyPath

    def __init__(self, cell: BaseCell, zmd: ZoneMapData,  zone_ind: int = 0):
        self.ns = "CellCoverageP-Z" + str(zone_ind)
        self.lg = MyLogger(self.ns)

        #
        self.cell = cell
        self.zmd = zmd

        # parameters and configurations
        self.pcfgs = PlanningConfigs()
        self.resolution = zmd.reslolution

        #
        self.path_m = None
        self.path_px = None

        #
        self.coverage_planning()

    def coverage_planning(self) -> None:
        if not self.cell.do_planning:
            self.lg.warn(f"cell {self.cell.id} does not require planning.")
            return
        # self.calculate_path_by_poly_buffer()
        self.calculate_path_by_distance_transform()
        self.create_path_m()

    def calculate_path_by_poly_buffer(self) -> None:
        poly = self.cell.poly
        # scale the polygon to smaller size

        robot_r_px = self.pcfgs.cleaning_r_ov / self.resolution
        robot_d_px = robot_r_px * 2

        scaled_polys = []
        points = []

        # scaled_poly = scale(poly, xfact=0.5, yfact=0.5, origin='center')

        #
        scaled_poly = poly.buffer(-1, join_style=2, resolution=8)  # initial scale

        #
        i = 1
        while not (scaled_poly.is_empty or scaled_poly is None):
            print(i)
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

    def calculate_path_by_distance_transform(self) -> None:

        #
        robot_r_px = int(self.pcfgs.cleaning_r_ov / self.resolution)
        robot_d_px = int(robot_r_px * 2)

        #
        points = []

        #
        poly = self.cell.poly
        scaled_poly = poly.buffer(-1, join_style=2, resolution=8)  # initial scale
        if scaled_poly.is_empty or scaled_poly is None:
            return
        if isinstance(scaled_poly, MultiPolygon):
            scaled_poly = scaled_poly.geoms[0]
        points.extend(list(scaled_poly.exterior.coords))

        s_x = self.zmd.rectangle.x_min
        s_y = self.zmd.rectangle.y_min
        s_xy = np.array([s_x, s_y], dtype=np.int32)

        # create a cell map
        z_map = self.zmd.map_ws.copy()
        c_map = np.zeros_like(z_map, dtype=np.uint8)
        cv.fillPoly(c_map, [np.array(self.cell.points)-s_xy], color=255)
        mask = c_map == 255
        c_map[mask] = z_map[mask]

        # todo could crop the c_map to the cell polygon map

        # cv.imshow("c_map", c_map)
        # cv.waitKey()
        # cv.destroyAllWindows()

        # distance transform
        dist_transform = cv.distanceTransform(c_map, cv.DIST_L2, 5)

        # max value in the distance transform
        curr_dist = 0
        max_dist = dist_transform.max()

        # find polygon for each distance
        while curr_dist < (max_dist-robot_d_px):
            curr_dist += robot_d_px

            polys = self.find_poly_by_dist(c_map, dist_transform, curr_dist)
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
            polys = self.find_poly_by_dist(c_map, dist_transform, max_dist)
            if polys is not None and len(polys) > 0:
                for poly in polys:
                    new_points = poly.exterior.coords + s_xy
                    # rearrange points starting from the closest point to the last point in points
                    new_points = self.rearrange_points(points[-1], new_points)
                    points.extend(list(new_points))

        # create a path from the scaled polygon
        points = [MyPoint(int(p[0]), int(p[1])) for p in points]
        self.path_px = MyPath(points)

    def find_poly_by_dist(self, c_map, dist_transform: np.ndarray, dist: int, tolerance: int = 1) -> list[Polygon]:

        # Binary mask of pixels close to target distance
        mask = (dist_transform >= dist - tolerance) & (dist_transform <= dist + tolerance)
        # mask = (dist_transform == dist)  # exact match for distance

        # Convert boolean mask to uint8
        mask = mask.astype(np.uint8) * 255

        # Optional: clean up mask (remove noise)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.lg.error(f"No contours found for cell {self.cell.id}.")
            return

        # # for cnt in contours:
        # for i, cnt in enumerate(contours):
        #     cv.drawContours(c_map, [cnt], -1, 100, 1)
        # c_map[mask] = 100
        # cv.imshow("c_map", c_map)
        # cv.waitKey()
        # cv.destroyAllWindows()

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

    def rearrange_polys_by_area(self, polys: list[Polygon]) -> list[Polygon]:
        """ Rearrange polygons by area in descending order """
        polys.sort(key=lambda p: p.area, reverse=True)
        return polys

    def rearrange_points(self, point: np.ndarray, points: np.ndarray) -> np.ndarray:
        """ Rearrange points starting from the closest point to the given point """
        # Find the index of the closest point to the given point
        idx = self.find_closest_point_idx(point, points)
        # Rearrange points starting from the closest point
        rearranged_points = self.rearrange_points_by_idx(idx, points)
        # add first point to the end of the list
        rearranged_points = np.concatenate((rearranged_points, [rearranged_points[0]]), axis=0)
        return rearranged_points

    def rearrange_points_by_idx(self, idx: int, points: np.ndarray) -> np.ndarray:
        """ Rearrange points starting from idx """
        if idx < 0 or idx >= len(points):
            self.lg.error(f"Index {idx} is out of bounds for points list.")
            return points
        return np.concatenate((points[idx:], points[:idx]), axis=0)

    def find_closest_point_idx(self, p: np.ndarray, points: np.ndarray) -> int:
        """ Find the index of the closest point to p in points """
        dists = [np.linalg.norm(np.array(p) - np.array(pt)) for pt in points]
        return np.argmin(dists)

    def create_path_m(self) -> None:
        if self.path_px:
            self.path_m = MyPath([self.point_px_to_m(p) for p in self.path_px.points])

    def point_px_to_m(self, p: MyPoint) -> MyPoint:
        return MyPoint(p.x * self.resolution, p.y * self.resolution)
