import os
import cv2 as cv
import numpy as np
from shapely import Polygon, Point
from scripts.district.zone_map_data import ZoneMapData
from scripts.district.district import DistrictData, ZoneData
from scripts.planning.base_coverage_pp import CoveragePPBase
CASE_DEC = "Dec1-Trapezoidal"  # "D2-Boustraphedon" # Dec1-Trapezoidal
if CASE_DEC == "Dec2-Boustraphedon":
    from scripts.decomposition.decomposition_2_bs_mh_f import TrapezoidalDecomposition as CellDecomposition
else:
    from scripts.decomposition.decomposition_1_tp_mh_f import TrapezoidalDecomposition as CellDecomposition


class DecompositionInterface:
    def __init__(self, zmd: ZoneMapData):
        self.ns = "Decomposition"
        print(f"[{self.ns}]: Step: decomposing zone {zmd.ind}")

        #
        self.decomposed = CellDecomposition()

        # settings
        self.dec_mid_dist = 10

        # set bounding polygon
        self.create_bounding(zmd)

        # set obstacle polygons
        self.create_obstacle_polygons(zmd)

        # initiate decomposition
        self.decomposed.initiate()

    def create_bounding(self, zmd: ZoneMapData):
        # todo
        # y_max = max(zmd.vertices, key=lambda x: x[1])[1]

        #
        contours, _ = cv.findContours(zmd.map_ws, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv.contourArea)

        # check mid distance
        filtered_c = self.filter_contour_points(biggest_contour)
        filtered_c.reverse()
        b_poly = Polygon([Point(p[0][0], p[0][1]) for p in filtered_c])

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
            #
            filtered_c = self.filter_contour_points(c)

            # get the polygon
            if len(filtered_c) < 3:
                print(f"[{self.ns}]: not enough points to create polygon")
                continue
            poly = Polygon([Point(p[0][0], p[0][1]) for p in filtered_c])
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

# =====================================================================================================================
# =====================================================================================================================


class CoveragePP(CoveragePPBase):
    """  DecCoveragePP  """

    def __init__(self, district: DistrictData):
        super().__init__(district)
        self.ns = "DecCoveragePP"
        print(f"[{self.ns}]: Step: coverage path planning")

        #
        self.decompositions: list[DecompositionInterface] = []

        # coverage path planning for all zones
        self.trap_dec_cpp()
        # self.save_decomposition()

    def trap_dec_cpp(self):
        print(f"[{self.ns}]: start trap_dec_cpp for all zones")

        # for each zone
        for zd in self.district.zone_datas:
            tpd = DecompositionInterface(zd.zone_map_data)
            self.decompositions.append(tpd)

    def save_decomposition(self):
        base_path = self.district.dmd.orig_map_data.bcfg.district_dir
        for i, zd in enumerate(self.district.zone_datas):
            dec_path = base_path + f"/Zone_{zd.ind}"
            dec_path = os.path.join(dec_path, CASE_DEC)
            self.decompositions[i].tpd.save_figs(dec_path)

    def coverage_planning(self):
        print(f"[{self.ns}]: simple coverage path planning")
        # for each zone # list[ZoneData]
        for zd in self.district.zone_datas:
            self.zone_covpp(zd)
            self.post_process_os_path(zd)

    def zone_covpp(self, zd: ZoneData):
        print(f"[{self.ns}]: cpp for zone [{zd.zone_map_data.ind}]")

        # margin rectangle
        margin = self.pcfgs.margin

        # effective cleaning diameter
        cleaning_d = self.cleaning_width - self.min_overlap_w   # cleaning diameter

        # zone rectangle
        rd = zd.zone_map_data.final_rect_data  # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        resol = zd.zone_map_data.reslolution

        # w, h in meters
        w_m = rd["w"] * resol
        h_m = rd["h"] * resol
        # xc = rd["xc"]
        # yc = rd["yc"]
        # rot = rd["rot"]
        #
        w_m = w_m - 2 * margin
        h_m = h_m - 2 * margin + cleaning_d

        # xy min max
        x_min = 0
        y_min = 0
        # x_max = w
        # y_max = h

        # calculate planning w, h, start and end points
        # _p: planning
        n_lines = np.ceil(h_m / cleaning_d)                     # number of lines
        cleaning_d = h_m / n_lines                              # width of each cleaning line
        # cleaning_r = cleaning_d / 2                             # cleaning radius
        # margin_x = cleaning_r
        margin_x = 0

        # xy min max planning
        x_min_p = x_min + margin + margin_x / 2  # * (np.sqrt(2)/2)
        y_min_p = y_min + margin + margin_x
        # x_max_p = w - margin - w_l_p / 2
        # y_max_p = h - margin - w_l_p / 2

        # initial point
        init_point = (x_min_p, y_min_p)
        n_points = int(n_lines * 2)

        # dy dy
        dx = w_m - margin_x / 2  # * (np.sqrt(2)/2)
        dy = cleaning_d

        base_path = [init_point]

        for i in range(1, n_points):
            if i % 2 == 0:
                new_x = base_path[-1][0]
                new_y = base_path[-1][1] + dy
            else:
                new_x = base_path[-1][0] + dx
                new_y = base_path[-1][1]
                dx = -1 * dx
            base_path.append((new_x, new_y))

        # # base_path = base_path
        # path_x = (xy[0] for xy in base_path)
        # path_y = (xy[1] for xy in base_path)

        # save path to zone data
        zd.os_path_m = base_path

    def post_process_os_path(self, zd: ZoneData):
        os_path = zd.os_path_m
        rect_data = zd.zone_map_data.final_rect_data
        rot = rect_data["rot"]
        tl = zd.zone_map_data.final_rectangle.tl
        resol = zd.zone_map_data.reslolution
        x_0, y_0 = tl[0]*resol, tl[1]*resol  # first vertex (from which rot is calculated)
        # rotate
        new_os_path = [self.rotate_point(p[0], p[1], rot) for p in os_path]
        # translate
        new_os_path = [(p[0] + x_0, p[1] + y_0) for p in new_os_path]
        # update
        zd.os_path_m = new_os_path

    def rotate_point(self, x, y, rotation):
        x_r = x * np.cos(rotation) - y * np.sin(rotation)
        y_r = x * np.sin(rotation) + y * np.cos(rotation)
        return (x_r, y_r)
