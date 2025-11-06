"""Polygon extraction module for workspace boundaries and obstacles."""
import cv2 as cv
import numpy as np
# from shapely import affinity
from shapely import Polygon, Point
from scripts.base.logger import MyLogger
from scripts.district.zone_map_data import ZoneMapData


class MapPolygons:
    """Extract and represent workspace boundary and obstacle polygons.

    This class processes binary workspace maps to identify and create
    polygon representations of the coverage boundary and internal obstacles.
    """

    def __init__(self, zmd: ZoneMapData, map_ws: np.ndarray = None, dec_mid_dist: int = 10):
        self.lg = MyLogger("MapPolygons")
        self.b_poly: Polygon = None  # bounding polygon
        self.obs_polys: list[Polygon] = []  # obstacle polygons

        # map workspace
        if map_ws is not None:
            self.map_ws = map_ws
        else:
            self.map_ws = zmd.map_ws

        # starting coordinates
        self.s_x: int = None
        self.s_y: int = None

        # settings
        self.dec_mid_dist = dec_mid_dist
        self.set_start_coord(zmd)

        # set bounding polygon
        self.create_bounding()

        # set obstacle polygons
        self.create_obstacle_polygons()

    def set_start_coord(self, zmd: ZoneMapData):
        """Set starting coordinates from zone map rectangle."""
        self.s_x = zmd.rectangle.x_min
        self.s_y = zmd.rectangle.y_min

    def create_bounding(self):
        """Extract workspace boundary polygon from binary map.

        Uses contour detection to find the largest contour representing
        the workspace boundary.
        """
        contours, _ = cv.findContours(self.map_ws, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv.contourArea)

        # check mid distance
        filtered_c = self.filter_contour_points(biggest_contour)
        filtered_c.reverse()  # reverse to have clockwise order

        # shift points by starting coordinates
        self.b_poly = Polygon([Point(self.s_x+p[0][0], self.s_y+p[0][1]) for p in filtered_c])

    def create_obstacle_polygons(self):
        """Extract obstacle polygons from binary map using contour detection."""
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
                self.lg.warning("Not enough points to create polygon for contour in zone")
                continue
            poly = Polygon([Point(self.s_x+p[0][0], self.s_y+p[0][1]) for p in filtered_c])
            obs_polys.append(poly)
        self.obs_polys = obs_polys

    def filter_contour_points(self, contour: np.ndarray):
        """ Filter contour points by minimum distance."""
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
        """ Rotate all polygons (bounding and obstacles) by the given angle around their centroid. """
        self.b_poly = self.rotate_polygon(self.b_poly, angle, origin_t)
        rotated_obs_polys = []
        for poly in self.obs_polys:
            rotated_poly = self.rotate_polygon(poly, angle, origin_t)
            rotated_obs_polys.append(rotated_poly)
        self.obs_polys = rotated_obs_polys

    def rotate_polygon(self, poly: Polygon, angle: float, origin_t: np.ndarray = None) -> Polygon:
        """ Rotate the given polygon by the given angle around the given origin. """
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
