import cv2 as cv
import numpy as np
from shapely import Polygon, Point
from scripts.district.zone_map_data import ZoneMapData
from scripts.boustrophedon.decomposition_1_mh_f import MHDecomposition as CellDecomposition


class DecompositionInterface:
    def __init__(self, zmd: ZoneMapData):
        self.ns = "Decomposition"
        self.decomposed = CellDecomposition()
        self.s_x: int = 0
        self.s_y: int = 0

        # settings
        self.dec_mid_dist = 10
        self.set_start_coord(zmd)

        # set bounding polygon
        self.create_bounding(zmd)

        # set obstacle polygons
        self.create_obstacle_polygons(zmd)

        # initiate decomposition
        self.decomposed.initiate()

    def set_start_coord(self, zmd: ZoneMapData):
        self.s_x = zmd.rectangle.x_min
        self.s_y = zmd.rectangle.y_min

    def create_bounding(self, zmd: ZoneMapData):
        contours, _ = cv.findContours(zmd.map_ws, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv.contourArea)

        # check mid distance
        filtered_c = self.filter_contour_points(biggest_contour)
        filtered_c.reverse()
        b_poly = Polygon([Point(self.s_x+p[0][0], self.s_y+p[0][1]) for p in filtered_c])

        self.decomposed.set_bounding_polygon(b_poly)

    def create_obstacle_polygons(self, zmd: ZoneMapData):
        # get obstacle contours inside the biggest contour from map imag
        contours, hierarchy = cv.findContours(zmd.map_ws, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        # get the biggest contour

        # get the obstacle contours
        obs_contours = []
        for i, element in enumerate(hierarchy[0]):
            if element[3] != -1:
                obs_contours.append(contours[i])

        #
        if len(obs_contours) == 0:
            print(f"[{self.ns}]: no obstacle contours found in zone {zmd.ind}")
            return

        # obstacle polygons
        obs_polys = []
        for c in obs_contours:
            filtered_c = self.filter_contour_points(c)
            # get the polygon
            if len(filtered_c) < 3:
                print(f"[{self.ns}]: not enough points to create polygon")
                continue
            poly = Polygon([Point(self.s_x+p[0][0], self.s_y+p[0][1]) for p in filtered_c])
            obs_polys.append(poly)

        # set obstacle polygons
        self.decomposed.set_obstacle_polygons(obs_polys)

    def filter_contour_points(self, contour: np.ndarray):
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
