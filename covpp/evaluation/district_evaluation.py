"""District-level evaluation for aggregating zone statistics."""
import json
from covpp.evaluation.zone_evaluation_base import ZoneStatsBase


class DistrictStatsBase:
    """Container for aggregated district-level KPIs.

    Result:
        Holds the sums/ratios derived from all zone statistics so they can be
        serialized once per planning run via :meth:`get_data_dict`.
    """

    def __init__(self):
        self.map_name: str = None
        self.method: str = None
        self.n_zones: int = None
        self.full_trav_time: float = None
        self.coverage_trav_time: float = None
        #
        self.time_segmentation: float = None
        self.time_coverage_planning: float = None
        self.time_sequencing: float = None
        self.time_total: float = None
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
        """Convert statistics to dictionary format for serialization."""
        return {
            "map_name": self.map_name,
            "method": self.method,
            "n_zones": int(self.n_zones),
            "full_trav_time": round(float(self.full_trav_time), 2),
            "coverage_trav_time": round(float(self.coverage_trav_time), 2),
            "time_segmentation": round(float(self.time_segmentation), 2),
            "time_coverage_planning": round(float(self.time_coverage_planning), 2),
            "time_sequencing": round(float(self.time_sequencing), 2),
            "time_total": round(float(self.time_total), 2),
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
    """Aggregate zone metrics into a single district report.

    Steps:
        1. Accept a list of ``ZoneStatsBase`` objects after every zone finishes.
        2. Sum compatible quantities, recompute ratios, and store them in
           ``self.stats``.
        3. Persist the aggregated report to ``result_file`` for later inspection.
    """

    def __init__(self, result_file: str = None):
        self.stats = DistrictStatsBase()
        self.result_file = result_file

    def evaluate(self, zone_stats: list[ZoneStatsBase]):
        """Aggregate zone statistics to compute district-level metrics."""
        self.stats.map_name = zone_stats[0].map_name
        self.stats.method = zone_stats[0].method
        self.stats.n_zones = len(zone_stats)

        if self.stats.n_zones == 0:
            raise ValueError("No zone statistics provided for district evaluation.")

        # aggregate statistics from all zones
        self.stats.full_trav_time = sum(zs.full_trav_time for zs in zone_stats)
        self.stats.coverage_trav_time = sum(zs.coverage_trav_time for zs in zone_stats)
        self.stats.time_segmentation = sum(zs.time_segmentation for zs in zone_stats)
        self.stats.time_coverage_planning = sum(zs.time_coverage_planning for zs in zone_stats)
        self.stats.time_sequencing = sum(zs.time_sequencing for zs in zone_stats)
        self.stats.time_total = sum(zs.time_total for zs in zone_stats)
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
        with open(self.result_file, 'wt', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
