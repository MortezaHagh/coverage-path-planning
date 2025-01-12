
""" All configs """


class BConfigs:
    """ base configs """

    def __init__(self):
        # map configs
        self.origin = None
        self.resolution = None
        self.free_thresh = None
        self.occupied_thresh = None

        # directions
        self.map_name = None
        self.map_path = None
        self.data_dir = None
        self.district_folder = "District"


class DistrictConfgs:
    """ district configs"""

    def __init__(self):
        self.whole_map = False  # False True


class SpaceConfigs:
    def __init__(self):
        # robot configs
        self.robot_radius = 0.5
        self.robot_circumscribed_r = 0.5
        self.robot_circumscribed_d = self.robot_circumscribed_r * 2
        #
        self.wf_safe_dist = 0.1  # wf distance
        self.os_safe_dist = 0.2  # safe distance of os path from obstacles

        self.wf_min_obst_dist = self.robot_radius + self.wf_safe_dist  # distance of the wf path from walls and obstacles
        self.os_min_obst_dist = self.robot_radius + self.os_safe_dist  # min distance of the os path from walls and obstacles
        self.free_space_obst_dist = round(max(self.wf_min_obst_dist, self.os_min_obst_dist), 2)
        #
        self.scaler_os_search = 2
        #
        self.black = 50
