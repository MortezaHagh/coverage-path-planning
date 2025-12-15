"""Polygon extraction for workspace boundaries and obstacles.

Steps:
- Detect external workspace boundary via contours
- Simplify contours by distance thresholding
- Build shapely polygons in global pixel coordinates
- Validate and fix polygons if needed

Result:
- ``b_poly`` for the boundary and ``obs_polys`` for obstacles.
"""
import cv2 as cv
import numpy as np
# from shapely import affinity
from shapely import Polygon, Point
from covpp.base.logger import MyLogger
from covpp.base.geometry import evaluate_polygon


class MapPolygons:
    """Extract and represent workspace boundary and obstacle polygons.

    This class processes binary workspace maps to identify and create
    polygon representations of the coverage boundary and internal obstacles.

    Args:
        map_ws (np.ndarray): Binary map of the workspace (1: free, 0: occupied).
        xy (np.ndarray): Starting coordinates (offset) in the global map.
        dec_mid_dist (int): Minimum distance to keep contour points.
    """

    def __init__(self, map_ws: np.ndarray, xy: np.ndarray, dec_mid_dist: int = 10):
        self.lg = MyLogger("MapPolygons")

        # polygons
        self.b_poly: Polygon = None         # bounding polygon (in global coordinates (pixels))
        self.obs_polys: list[Polygon] = []  # obstacle polygons (in global coordinates (pixels))
        self.success: bool = True

        # map workspace
        self.map_ws = map_ws

        # starting coordinates (offset in global map)
        self.s_x: int = xy[0]
        self.s_y: int = xy[1]

        # settings
        self.dec_mid_dist = dec_mid_dist  # min distance to keep contour points

        # set bounding polygon
        if not self.create_bounding():
            self.success = False
            return

        # set obstacle polygons
        self.create_obstacle_polygons()

    def create_bounding(self) -> bool:
        """Extract workspace boundary polygon from binary map.

        Steps:
        - Find largest external contour as the workspace boundary
        - Filter contour points by minimum distance
        - Convert to global pixel coordinates and validate polygon

        Returns:
        - bool: True on success; False when boundary creation fails.
        """
        contours, _ = cv.findContours(self.map_ws, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv.contourArea)

        # check mid distance
        filtered_c = self.filter_contour_points(biggest_contour)
        filtered_c.reverse()  # reverse to have clockwise order

        # check size
        if len(filtered_c) < 3:
            self.lg.warn("Not enough points to create bounding polygon.")
            return False

        # shift points by starting coordinates
        self.b_poly = Polygon([Point(self.s_x+p[0][0], self.s_y+p[0][1]) for p in filtered_c])

        # check validity
        is_valid, status = evaluate_polygon(self.b_poly)
        if not is_valid:
            self.lg.warn(f"Bounding polygon is not valid, Status: {status}")
            return False

        return True

    def create_obstacle_polygons(self):
        """Extract obstacle polygons inside the workspace boundary.

        Steps:
        - Find inner contours (holes) of the workspace
        - Filter, convert to global coordinates, and validate polygons
        - Attempt fix via ``buffer(0)`` when invalid
        """
        # get obstacle contours inside the biggest contour from map image
        contours, hierarchy = cv.findContours(self.map_ws, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        # get the obstacle contours
        obs_contours = []
        for i, element in enumerate(hierarchy[0]):
            if element[3] != -1:
                obs_contours.append(contours[i])

        # check if there are any obstacle contours
        if len(obs_contours) == 0:
            # self.lg.info(f"No obstacle contours found in zone {zmd.ind}")
            return

        # obstacle polygons
        obs_polys = []
        for c in obs_contours:
            filtered_c = self.filter_contour_points(c)
            # get the polygon
            if len(filtered_c) < 3:
                self.lg.warn("Not enough points to create polygon for an obstacle, skipping.")
                continue
            poly = Polygon([Point(self.s_x+p[0][0], self.s_y+p[0][1]) for p in filtered_c])

            # check validity
            is_valid, status = evaluate_polygon(poly)
            if not is_valid:
                self.lg.warn(f"Obstacle polygon is not valid, Status: {status}. Trying to fix it.")
                poly = poly.buffer(0)
                is_valid, status = evaluate_polygon(poly)
                if not is_valid:
                    self.lg.warn(f"Obstacle polygon is still not valid after fixing, skipping. Status: {status}")
                    raise ValueError(f"Obstacle polygon is still not valid after fixing. Status: {status}")
            obs_polys.append(poly)
        self.obs_polys = obs_polys

    def filter_contour_points(self, contour: np.ndarray):
        """Filter contour points by a minimum Euclidean distance.

        Returns:
        - list[np.ndarray]: Subset of contour points satisfying spacing.
        """
        skip = set()
        for i, p1 in enumerate(contour):
            if i in skip:
                continue
            for j in range(i+1, len(contour)):
                if i in skip:
                    continue
                dist = np.linalg.norm(p1 - contour[j])
                if dist < self.dec_mid_dist:
                    skip.add(j)
        filtered_c = [contour[i] for i in range(len(contour)) if i not in skip]
        return filtered_c

    def rotate_all_polygons(self, angle: float, origin_t: np.ndarray = None) -> None:
        """Rotate boundary and obstacle polygons around an origin.

        Args:
        - angle: Rotation angle in radians.
        - origin_t: Optional origin ``[x, y]``; defaults to each polygon centroid.
        """
        self.b_poly = self.rotate_polygon(self.b_poly, angle, origin_t)
        rotated_obs_polys = []
        for poly in self.obs_polys:
            rotated_poly = self.rotate_polygon(poly, angle, origin_t)
            rotated_obs_polys.append(rotated_poly)
        self.obs_polys = rotated_obs_polys

    def rotate_polygon(self, poly: Polygon, angle: float, origin_t: np.ndarray = None) -> Polygon:
        """Rotate a polygon by an angle around an origin.

        Args:
        - poly: Input polygon.
        - angle: Rotation angle in radians.
        - origin_t: Optional origin ``[x, y]``; defaults to polygon centroid.

        Returns:
        - Polygon: Rotated polygon (not re-shifted back to global origin).
        """
        # rotate bounding polygon
        if origin_t is None:
            origin = np.array(poly.centroid.coords)[0]
        else:
            origin = origin_t

        # translate to origin
        xy = np.array(poly.exterior.coords) - origin
        # rotate
        rx = (xy[:, 0] * np.cos(angle) - xy[:, 1] * np.sin(angle))
        ry = (xy[:, 0] * np.sin(angle) + xy[:, 1] * np.cos(angle))
        rxy = np.column_stack((rx, ry))
        new_poly = Polygon(rxy)
        return new_poly

    # def rotate_all_polygons(self, angle: float, origin_t: tuple = (0, 0)) -> None:
    #     """ Rotate all polygons (bounding and obstacles) by the given angle around their centroid. """

    #     angle_deg = np.degrees(angle)
    #     # origin = 'centroid'
    #     origin = Point(origin_t)

    #     # rotate bounding polygon
    #     if self.b_poly is not None:
    #         self.b_poly = affinity.rotate(self.b_poly, angle_deg, origin=origin)

    #     # rotate obstacle polygons
    #     rotated_obs_polys = []
    #     for poly in self.obs_polys:
    #         rotated_poly = affinity.rotate(poly, angle_deg, origin=origin)
    #         rotated_obs_polys.append(rotated_poly)
    #     self.obs_polys = rotated_obs_polys
