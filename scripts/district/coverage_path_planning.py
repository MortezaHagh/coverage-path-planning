import numpy as np
from scripts.base.configs import SpaceConfigs, PlanningConfigs
from scripts.district.district import DistrictData, ZoneData


class CPPSimple:
    district: DistrictData  # main district data

    def __init__(self, district: DistrictData):
        self.ns = "CPPSimple"
        print(f"[{self.ns}]: Step: coverage path planning")

        # main district data
        self.district = district

        # # parameters and configurations
        self.scfgs = SpaceConfigs()
        self.pcfgs = PlanningConfigs()

        # ---------------------------------------------
        # robot params
        self.robot_width = self.scfgs.robot_radius
        self.min_overlap_w = self.pcfgs.min_path_overlap
        self.cleaning_width = 2*self.scfgs.cleaning_r
        # map params
        self.resolution = district.dmd.reslolution
        # planning configs
        self.min_points_distance = self.pcfgs.min_wp_dist
        # ---------------------------------------------

        # coverage path planning for all zones
        self.cpp_simple()

    def cpp_simple(self):
        print(f"[{self.ns}]: simple coverage path planning")

        # for each zone # list[ZoneData]
        for zd in self.district.zone_datas:
            self.zone_cpp(zd)
            self.post_process_os_path(zd)

    def zone_cpp(self, zd: ZoneData):
        print(f"[{self.ns}]: cpp for zone [{zd.zone_map_data.ind}]")

        # margin rectangle
        margin = self.pcfgs.margin

        # effective cleaning diameter
        cleaning_d = self.cleaning_width - self.min_overlap_w   # cleaning diameter

        # zone rectangle
        rd = zd.zone_map_data.rect_data  # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        resol = zd.zone_map_data.reslolution

        # w, h in meters
        w_m = rd["w"] * resol
        h_m = rd["h"] * resol
        # xc = rd["xc"]
        # yc = rd["yc"]
        # rot = rd["rot"]
        #
        w_m = w_m - 2 * margin
        h_m = h_m - 2 * margin

        # xy min max
        x_min = 0
        y_min = 0
        # x_max = w
        # y_max = h

        # calculate planning w, h, start and end points
        # _p: planning
        n_lines = np.ceil(h_m / cleaning_d)                     # number of lines
        cleaning_d = h_m / n_lines                              # width of each cleaning line

        # xy min max planning
        x_min_p = x_min + margin + cleaning_d / 2
        y_min_p = y_min + margin + cleaning_d / 2
        # x_max_p = w - margin - w_l_p / 2
        # y_max_p = h - margin - w_l_p / 2

        # initial point
        init_point = (x_min_p, y_min_p)
        n_points = int(n_lines * 2)

        # dy dy
        dx = w_m - cleaning_d / 2
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

        # base_path = base_path
        path_x = (xy[0] for xy in base_path)
        path_y = (xy[1] for xy in base_path)

        zd.os_path_m = base_path

        # todo
        # change coordinate frame to zone
        # translation and rotation of the rectangle

    def post_process_os_path(self, zd: ZoneData):
        os_path = zd.os_path_m
        rect_data = zd.zone_map_data.rect_data
        rot = rect_data["rot"]
        tl = zd.zone_map_data.rectangle.tl
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
