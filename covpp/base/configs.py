
"""Configuration dataclasses used by the pipeline."""

import dataclasses


@dataclasses.dataclass
class MapConfig:
    """Container for occupancy grid metadata and map directory layout.

    Attributes capture the transform from pixel to metric space in addition to
    the on-disk structure of a dataset (e.g. base, district folders).
    """
    # map configs
    origin_m = None         # origin x, y in meters
    resolution = None       # meters/pixel
    free_thresh = None      # [not used] Maximum occupancy probability below which a cell is considered free
    occupied_thresh = None  # [not used] Minimum occupancy probability above which a cell is considered occupied
    # directories and paths
    data_dir = None         # base directory for maps ./results/
    map_name = None         # name of the main map folder
    base_folder = "AMain"   # inside each map folder, contains main map files
    district_folder = "District"
    map_dir = None          # path to main map folder: ./results/<map_name>/AMain
    district_dir = None     # path to district folder: ./results/<map_name>/District


@dataclasses.dataclass
class PipeLineConfig:
    """Global switches that describe the end-to-end coverage pipeline setup.

    The configuration controls which segmentation, coverage path planning, and
    sequencing modules are used together with plotting and debugging options.
    """

    # Main configs
    whole_map = True    # Select whole map as district (True) or select rectangle (False)
    method = ""         # full method name (to be completed based on other configs)
    start_cell_ind = 0  # index of starting cell for coverage path planning sequence

    # plots configs
    save_map_planning = True
    save_decomposition_plot = True
    plot_decomposition = False
    dpi = 300

    # Methods names for main modules: segmentation, coverage path planning, sequence
    segmentation_method = "voronoi"     # simple, voronoi
    covpp_method = "boust"            # simple, boust, contour, grid_energy_min
    sequence_method = "milp"            # simple, dfs, milp

    # complete method
    method = segmentation_method + '_' + covpp_method + '_' + sequence_method

    # segmentation options
    use_insc_ret = False  # find approx rectangles from initial segmentation cells

    # voronoi specific options
    if segmentation_method == "voronoi":
        if use_insc_ret:
            method = method + "_rect"

    @classmethod
    def update(cls, segmentation_method: str = 'voronoi', covpp_method: str = 'boust',
               use_insc_ret: bool = False, sequence_method: str = 'milp', start_cell_idx: int = 0) -> None:
        """Update pipeline configuration values."""
        cls.segmentation_method = segmentation_method
        cls.covpp_method = covpp_method
        cls.sequence_method = sequence_method
        cls.use_insc_ret = use_insc_ret
        cls.start_cell_ind = start_cell_idx

        # update main method
        cls.method = cls.segmentation_method + '_' + cls.covpp_method + '_' + cls.sequence_method

        # voronoi method
        if segmentation_method == "voronoi":
            if use_insc_ret:
                cls.method = cls.method + "_rect"


@dataclasses.dataclass
class RobotConfig:
    """Nominal robot characteristics used to estimate motion effort."""

    radius = 0.5
    v_lin_avg = 0.4  # average linear velocity in m/s
    v_ang_avg = 0.6  # average angular velocity in rad/s


@dataclasses.dataclass
class SpaceConfig:
    """Workspace safety margins and derived robot clearances."""

    select_free_space = False   # manually select free space area from image, otherwise use select biggest free space automatically

    # robot configs
    robot_radius = 0.5          # 0.5
    robot_radius_ws = 0.5       # 0.5

    # safety distance (robot footprint from obstacles)
    wf_safe_dist = 0.0  # 0.1                   # safety distance of robot during wall following from walls or obstacles
    os_safe_dist = 0.1  # 0.2                   # safe distance of robot during open space from obstacles
    ws_dist = max(wf_safe_dist, os_safe_dist)   # workspace safety distance: max required distance of robot from obstacles

    # safety distances (robot center from obstacles)
    wf_r_dist = robot_radius_ws + wf_safe_dist  # safety distance of robot wall following path from walls and obstacles
    os_r_dist = robot_radius_ws + os_safe_dist  # safety distance of robot open space path from obstacles
    ws_r_dist = ws_dist + robot_radius_ws       # workspace safety distance: max required distance of robot center from obstacles

    # cleaning radius
    cleaning_c = 1.0                            # effective cleaning coefficient
    cleaning_r = robot_radius * cleaning_c      # effective cleaning radius

    # remained uncleaned distance due to safety distance and effective cleaning radius
    uncleaned_dist = ws_dist + (robot_radius - cleaning_r)

    # other
    scaler_os_search = 2


@dataclasses.dataclass
class PlanningConfig:
    """Aggregated planning parameters derived from space analysis inputs."""
    spa = SpaceConfig()
    animation = False                                   # animate final coverage path in sequence planning
    cleaning_r = spa.cleaning_r                         # effective cleaning radius
    min_path_overlap = 0.0                              # minimum overlap required between adjacent cleaning paths
    margin = 0.05                                       # margin from cell boundaries for coverage planning
    cleaning_r_ov = cleaning_r - min_path_overlap/2     # effective cleaning radius considering path overlap
    min_wp_dist = cleaning_r_ov/2                       # minimum distance between waypoints in coverage path
