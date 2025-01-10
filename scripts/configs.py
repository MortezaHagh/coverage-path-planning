
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
        self.robot_circumscribed_d = 1.0
        self.wf_safe_dist = 0.1  # wf distance
        self.os_safe_dist = 0.3
        # distance of the wall following path from walls and obstacles
        self.wf_min_obst_dist = self.robot_radius + self.wf_safe_dist
        # min distance of the open space path from walls and obstacles
        self.os_min_obst_dist = self.robot_radius + self.os_safe_dist
        self.black = 50
        self.wall_dist_prescaler = 2.0
