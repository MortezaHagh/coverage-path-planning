import numpy as np
from scripts.base.configs import SpaceConfigs, PlanningConfigs
from scripts.district.district import DistrictData, ZoneData


class CoveragePPBase:
    district: DistrictData  # main district data

    def __init__(self, district: DistrictData):
        self.ns = "CoveragePPBase"
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
