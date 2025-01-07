
""" All configs """


class BConfigs:
    """ base configs """

    def __init__(self):
        # district
        self.whole_map = True  # False True

        # robot configs
        self.robot_radius = 0.5
        self.safe_distance = 0.2

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
        self.whole_map = False
