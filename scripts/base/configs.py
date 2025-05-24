
""" All configs """

import dataclasses


@dataclasses.dataclass
class BConfigs:
    def __init__(self):
        # map configs
        self.origin_m = None
        self.resolution = None
        self.free_thresh = None
        self.occupied_thresh = None
        # directions
        self.map_name = None
        self.map_path = None
        self.data_dir = None
        self.district_dir = None
        self.district_folder = "District"
        self.base_folder = "AMain"


@dataclasses.dataclass
class DistrictConfgs:
    def __init__(self):
        self.whole_map = True  # False True
        self.ns = ""

        # plots configs
        self.save_map_planning = True
        self.save_decomposition_plot = False
        self.plot_decomposition = False

        # coverage path planning configs
        self.covpp_method = "m1_boust"  # m0_simple, m1_boust

        if self.covpp_method == "m0_simple":
            pass
        elif self.covpp_method == "m1_boust":
            # boustrophedon configs
            self.cell_sequence_method = "s2_dfs"  # s1_simple, s2_dfs, s3_milp
            self.ns = self.cell_sequence_method


@dataclasses.dataclass
class SpaceConfigs:
    def __init__(self):
        # robot configs
        self.robot_radius = 0.5
        # self.robot_circumscribed_r = 0.5
        # self.robot_circumscribed_d = self.robot_circumscribed_r * 2
        # distance thresholds
        self.wf_safe_dist = 0.1  # wf distance
        self.os_safe_dist = 0.2  # safe distance of os path from obstacles
        self.ws_dist = max(self.wf_safe_dist, self.os_safe_dist)
        #
        self.wf_r_dist = self.robot_radius + self.wf_safe_dist  # distance of the wf path from walls and obstacles
        self.os_r_dist = self.robot_radius + self.os_safe_dist  # min distance of the os path from walls and obstacles
        self.ws_r_dist = self.ws_dist + self.robot_radius
        #
        self.cleaning_c = 1.0
        self.cleaning_r = self.robot_radius * self.cleaning_c
        self.uncleaned_dist = self.ws_dist + (self.robot_radius - self.cleaning_r)
        # other
        self.scaler_os_search = 2
        self.black = 50


@dataclasses.dataclass
class PlanningConfigs:
    def __init__(self):
        spa = SpaceConfigs()
        self.cleaning_r = spa.cleaning_r
        self.min_path_overlap = 0.0
        self.min_wp_dist = 0.4
        self.margin = 0.05
        self.cleaning_r_ov = self.cleaning_r - self.min_path_overlap/2


if __name__ == "__main__":
    s_cfgs = SpaceConfigs()
    print("ws_r_dist", s_cfgs.ws_r_dist)
    print("cleaning_dist", s_cfgs.uncleaned_dist)
    print("cleaning_r", s_cfgs.cleaning_r)
