import json
from scripts.base.logger import MyLogger
from scripts.sequence.sequence import SequencePaths
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import DistrictConfigs, PlanningConfigs


class ZoneStatsBase:
    """Zone statistics Base class to store zone statistics. 
    Attributes:
        full_length (float): full path length in meters (coverage and connecting)
        coverage_length (float): coverage path length in meters (todo)
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
        processing_time (float): processing time in seconds
    """

    def __init__(self):
        self.full_length = None
        self.coverage_length = None
        self.full_yaw_rad = None
        self.coverage_yaw_rad = None
        self.cleaning_area_px = None
        self.cleaning_area_m = None
        self.cleaned_area_px = 0
        self.cleaned_area_m = 0
        self.uncleaned_area_m = None
        self.overlaped_area_m = None
        self.coverage_p = None
        self.overlap_p = None
        self.processing_time = 0.0

    def get_data_dict(self) -> dict:
        """Get statistics data as a dictionary. """
        return {
            "full_length": float(self.full_length),
            "coverage_length": float(self.coverage_length),
            "full_yaw_rad": float(self.full_yaw_rad),
            "coverage_yaw_rad": float(self.coverage_yaw_rad),
            "cleaning_area_px": int(self.cleaning_area_px),
            "cleaning_area_m": float(self.cleaning_area_m),
            "cleaned_area_px": int(self.cleaned_area_px),
            "cleaned_area_m": float(self.cleaned_area_m),
            "uncleaned_area_m": float(self.uncleaned_area_m),
            "overlaped_area_m": float(self.overlaped_area_m),
            "coverage_p": float(self.coverage_p),
            "overlap_p": float(self.overlap_p),
            "processing_time": float(self.processing_time)
        }

    def print_stats(self):
        print("[Results]: ******************************************************* ")
        print(f"[Results]: cleaning area: {self.cleaning_area_m:.2f} m^2")
        print(f"[Results]: cleaned area: {self.cleaned_area_m:.2f} m^2")


class ZoneEvaluationBase:
    """Zone evaluation Base class to evaluate zone statistics.
    """

    def __init__(self, zmd: ZoneMapData, paths: SequencePaths):
        self.lg = MyLogger("ZoneEvaluationBase")
        self.zmd: ZoneMapData = zmd
        self.paths: SequencePaths = paths
        self.dcfgs = DistrictConfigs()
        self.pcfgs = PlanningConfigs()
        # statistics
        self.stats: ZoneStatsBase = None

    def evaluate(self, duration: float = 0.0):
        """ evaluate zone statistics. To be implemented in derived classes. """
        self.stats = ZoneStatsBase()
        self.save_result()

    def save_result(self):
        """ save zone statistics and planner statistics of the zone to json file. """
        # save zone statistics to json file
        zone_stats = self.stats.get_data_dict()
        result = {
            "map": self.zmd.orig_map_data.name,
            "zone_ind": self.zmd.ind,
            "method": self.dcfgs.method,
            "zone_stats": zone_stats,
        }
        # rewrite result file
        with open(self.zmd.result_file, 'wt') as f:
            json.dump(result, f, indent=4)
