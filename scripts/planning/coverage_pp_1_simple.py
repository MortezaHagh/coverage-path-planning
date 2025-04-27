import cv2 as cv
import numpy as np
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.base_coverage_pp import CoveragePPBase


class CoveragePP(CoveragePPBase):
    """ CPPSimple"""

    def __init__(self, zone_map_data: ZoneMapData):
        super().__init__(zone_map_data)
        self.ns = "CPPSimple" + "-" + str(self.zmd.ind)
        print(f"[{self.ns}]: >>>>> Step: coverage path planning")

        # coverage path planning for all zones
        self.coverage_planning()

    def coverage_planning(self):
        print(f"[{self.ns}]: simple coverage path planning")
        self.zone_covpp()
        self.rotate_waypoints()
        self.create_path_px()
        self.add_path_to_map_planning()

    # =================================================================================================================

    def zone_covpp(self):
        print(f"[{self.ns}]: cpp for zone [{self.zmd.ind}]")

        # margin rectangle
        margin = self.pcfgs.margin

        # effective cleaning diameter
        cleaning_d = self.cleaning_width - self.min_overlap_w   # cleaning diameter

        # zone rectangle
        rd = self.zmd.final_rect_data  # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        resol = self.zmd.reslolution

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
        n_lines = np.ceil(h_m / cleaning_d)     # number of lines
        cleaning_d = h_m / n_lines              # width of each cleaning line
        # cleaning_r = cleaning_d / 2           # cleaning radius
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

        # save path
        self.path_m = base_path

    def add_path_to_map_planning(self):
        path = self.cov_planner.path_px
        for (p1, p2) in zip(path[:-1], path[1:]):
            self.zmd.map_planning = cv.line(self.zmd.map_planning, p1, p2, (0, 255, 10), 2)

        # vis robot
        alpha = 0.2
        pose = self.point_m_to_px(path[0])
        radius_px_1 = int(np.ceil(self.scfgs.ws_r_dist / self.zmd.orig_map_data.up_resolution))
        radius_px_2 = int(np.ceil(self.scfgs.robot_radius / self.zmd.orig_map_data.up_resolution))
        radius_px_3 = int(np.ceil(self.scfgs.cleaning_r / self.zmd.orig_map_data.up_resolution))
        overlay = self.zmd.map_planning.copy()
        cv.circle(overlay, pose, radius_px_1, [200, 200, 250], -1)
        cv.circle(overlay, pose, radius_px_2, [0, 0, 250], -1)
        cv.circle(overlay, pose, radius_px_3, [100, 0, 100], -1)
        cv.circle(overlay, pose, 5, [250, 250, 250], -1)
        self.zmd.map_planning = cv.addWeighted(overlay, alpha, self.zmd.map_planning, 1 - alpha, 0)
