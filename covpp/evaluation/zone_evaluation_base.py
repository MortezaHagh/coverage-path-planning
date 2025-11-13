import json
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.sequence_handler import SequencePaths
from covpp.base.configs import PipeLineConfig, PlanningConfig, RobotConfig


class ZoneStatsBase:
    """Zone statistics Base class to store zone statistics.
    Attributes:
        map_name (str): map name
        method (str): planning method
        ind (int): zone index
        full_trav_time (float): travel time in seconds
        coverage_trav_time (float): travel time in seconds (coverage path)
        time_segmentation (float): segmentation time in seconds
        time_coverage_planning (float): coverage planning time in seconds
        time_sequencing (float): sequencing time in seconds
        time_total (float): total processing time in seconds
        full_length (float): full path length in meters (coverage and connecting)
        coverage_length (float): coverage path length in meters
        full_yaw_rad (float): yaw changes in radians
        coverage_yaw_rad (float): yaw changes in radians (coverage path)
        cleaning_area_px (int): area to be cleaned in pixels
        cleaning_area_m (float): area to be cleaned in square meters
        cleaned_area_px (int): cleaned area in pixels
        cleaned_area_m (float): cleaned area in square meters
        uncleaned_area_m (float): uncleaned area in square meters
        overlaped_area_m (float): overlapped area in square meters
        coverage_p (float): coverage percentage
        overlap_p (float): overlap percentage
    """

    def __init__(self):
        self.map_name: str = None
        self.method: str = None
        self.ind: int = None  # zone index
        self.full_trav_time: float = 0.0
        self.coverage_trav_time: float = 0.0
        #
        self.time_segmentation: float = None
        self.time_coverage_planning: float = None
        self.time_sequencing: float = None
        self.time_total: float = None
        #
        self.full_length = None
        self.coverage_length = None
        self.full_yaw_rad = None
        self.coverage_yaw_rad = None
        #
        self.cleaning_area_px = None
        self.cleaning_area_m = None
        self.cleaned_area_px = 0
        self.cleaned_area_m = 0
        self.uncleaned_area_m = None
        self.overlaped_area_m = None
        self.coverage_p = None
        self.overlap_p = None

    def get_data_dict(self) -> dict:
        """Get statistics data as a dictionary. """
        return {
            "map_name": self.map_name,
            "method": self.method,
            "zone_ind": self.ind,
            "full_trav_time": round(float(self.full_trav_time), 2),
            "coverage_trav_time": round(float(self.coverage_trav_time), 2),
            "time_segmentation": round(float(self.time_segmentation), 2),
            "time_coverage_planning": round(float(self.time_coverage_planning), 2),
            "time_sequencing": round(float(self.time_sequencing), 2),
            "time_total": round(float(self.time_total), 2),
            "full_length": round(float(self.full_length), 2),
            "coverage_length": round(float(self.coverage_length), 2),
            "full_yaw_rad": round(float(self.full_yaw_rad), 2),
            "coverage_yaw_rad": round(float(self.coverage_yaw_rad), 2),
            "cleaning_area_px": int(self.cleaning_area_px),
            "cleaning_area_m": round(float(self.cleaning_area_m), 2),
            "cleaned_area_px": int(self.cleaned_area_px),
            "cleaned_area_m": round(float(self.cleaned_area_m), 2),
            "uncleaned_area_m": round(float(self.uncleaned_area_m), 2),
            "overlaped_area_m": round(float(self.overlaped_area_m), 2),
            "coverage_p": round(float(self.coverage_p), 2),
            "overlap_p": round(float(self.overlap_p), 2)
        }


class ZoneEvaluationBase:
    """Zone evaluation Base class to evaluate zone statistics.
    """

    def __init__(self, zmd: ZoneMapData, paths: SequencePaths):
        self.zmd: ZoneMapData = zmd
        self.paths: SequencePaths = paths
        self.dcfg = PipeLineConfig()
        self.pcfg = PlanningConfig()
        self.robot_cfgs = RobotConfig()
        # statistics
        self.stats: ZoneStatsBase = None

    def evaluate(self, times: float = 0.0):
        """ evaluate zone statistics. To be implemented in derived classes. """
        raise NotImplementedError("Zone evaluation subclasses must implement evaluate() method")

    def save_result(self):
        """ save zone statistics and planner statistics of the zone to json file. """
        zone_stats = self.stats.get_data_dict()
        with open(self.zmd.result_file, 'wt') as f:
            json.dump(zone_stats, f, indent=4)
