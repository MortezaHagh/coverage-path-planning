
import json
from scripts.evaluation.zone_evaluation_base import ZoneStatsBase


class DistrictStatsBase:
    """Base class for district statistics.
    """

    def __init__(self):
        self.map_name: str = None
        self.method: str = None
        self.n_zones: int = None
        self.processing_time: float = None
        #
        self.full_path_length: float = None
        self.coverage_path_length: float = None
        self.full_yaw_rad: float = None
        self.coverage_yaw_rad: float = None
        #
        self.cleaning_area_m: float = None
        self.cleaned_area_m: float = None
        self.uncleaned_area_m: float = None
        self.overlaped_area_m: float = None
        self.coverage_p: float = None
        self.overlap_p: float = None

    def get_data_dict(self) -> dict:
        """Get statistics data as a dictionary. """
        return {
            "map_name": self.map_name,
            "method": self.method,
            "n_zones": int(self.n_zones),
            "processing_time": round(float(self.processing_time), 2),
            "full_path_length": round(float(self.full_path_length), 2),
            "coverage_path_length": round(float(self.coverage_path_length), 2),
            "full_yaw_rad": round(float(self.full_yaw_rad), 2),
            "coverage_yaw_rad": round(float(self.coverage_yaw_rad), 2),
            "cleaning_area_m": round(float(self.cleaning_area_m), 2),
            "cleaned_area_m": round(float(self.cleaned_area_m), 2),
            "uncleaned_area_m": round(float(self.uncleaned_area_m), 2),
            "overlaped_area_m": round(float(self.overlaped_area_m), 2),
            "coverage_p": round(float(self.coverage_p), 2),
            "overlap_p": round(float(self.overlap_p), 2)
        }


class DistrictEvaluationHandler:
    """District evaluation handler class to manage district evaluations.
    """

    def __init__(self, result_file: str = None):
        self.stats = DistrictStatsBase()
        self.result_file = result_file

    def evaluate(self, zone_stats: list[ZoneStatsBase]):
        self.stats.map_name = zone_stats[0].map_name
        self.stats.method = zone_stats[0].method
        self.stats.n_zones = len(zone_stats)

        if self.stats.n_zones == 0:
            raise ValueError("No zone statistics provided for district evaluation.")

        # aggregate statistics from all zones
        self.stats.processing_time = sum(zs.processing_time for zs in zone_stats)
        self.stats.full_path_length = sum(zs.full_length for zs in zone_stats)
        self.stats.coverage_path_length = sum(zs.coverage_length for zs in zone_stats)
        self.stats.full_yaw_rad = sum(zs.full_yaw_rad for zs in zone_stats)
        self.stats.coverage_yaw_rad = sum(zs.coverage_yaw_rad for zs in zone_stats)
        self.stats.cleaning_area_m = sum(zs.cleaning_area_m for zs in zone_stats)
        self.stats.cleaned_area_m = sum(zs.cleaned_area_m for zs in zone_stats)
        self.stats.uncleaned_area_m = sum(zs.uncleaned_area_m for zs in zone_stats)
        self.stats.overlaped_area_m = sum(zs.overlaped_area_m for zs in zone_stats)
        self.stats.coverage_p = self.stats.cleaned_area_m / self.stats.cleaning_area_m * 100
        self.stats.overlap_p = self.stats.overlaped_area_m / self.stats.cleaning_area_m * 100

        # save result
        self.save_result()

    def save_result(self):
        """ save district statistics to json file. """
        results = self.stats.get_data_dict()
        with open(self.result_file, 'wt') as f:
            json.dump(results, f, indent=4)
