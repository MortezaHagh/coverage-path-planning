
""" All configs """

import dataclasses


@dataclasses.dataclass
class BConfigs:
    # map configs
    origin_m = None
    resolution = None
    free_thresh = None
    occupied_thresh = None
    # directions
    map_name = None
    map_path = None
    data_dir = None
    district_dir = None
    district_folder = "District"
    base_folder = "AMain"


@dataclasses.dataclass
class DistrictConfgs:
    whole_map = True  # False True
    ns = ""
    method = ""

    # plots configs
    save_map_planning = True
    save_decomposition_plot = True
    plot_decomposition = False

    # coverage path planning configs
    cell_sequence_method = "s3_milp"  # s1_simple, s2_dfs, s3_milp

    covpp_method = "m3_contour"  # m1_simple, m2_boust, m3_contour
    if covpp_method == "m2_boust":
        method = covpp_method + "_" + cell_sequence_method
    else:
        method = covpp_method


@dataclasses.dataclass
class SpaceConfigs:
    select_free_space = False  # select free space from image
    # robot configs
    robot_radius = 0.5  # 0.5
    # robot_circumscribed_r = 0.5
    # robot_circumscribed_d = robot_circumscribed_r * 2
    # distance thresholds
    wf_safe_dist = 0.0  # 0.1 # wf distance
    os_safe_dist = 0.1  # 0.2 # safe distance of os path from obstacles
    ws_dist = max(wf_safe_dist, os_safe_dist)
    #
    wf_r_dist = robot_radius + wf_safe_dist  # distance of the wf path from walls and obstacles
    os_r_dist = robot_radius + os_safe_dist  # min distance of the os path from walls and obstacles
    ws_r_dist = ws_dist + robot_radius
    #
    cleaning_c = 1.0
    cleaning_r = robot_radius * cleaning_c
    uncleaned_dist = ws_dist + (robot_radius - cleaning_r)
    # other
    scaler_os_search = 2
    black = 50


@dataclasses.dataclass
class PlanningConfigs:
    spa = SpaceConfigs()
    animation = False  # True False
    cleaning_r = spa.cleaning_r
    min_path_overlap = 0.0
    min_wp_dist = 0.4
    margin = 0.05
    cleaning_r_ov = cleaning_r - min_path_overlap/2


if __name__ == "__main__":
    s_cfgs = SpaceConfigs()
    print("ws_r_dist", s_cfgs.ws_r_dist)
    print("cleaning_dist", s_cfgs.uncleaned_dist)
    print("cleaning_r", s_cfgs.cleaning_r)
