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
        self.n_cells: int = None
        self.n_cov_cells: int = None
        self.trav_time_full: float = None
        self.trav_time_coverage: float = None

        #
        self.ptime_segmentation: float = None
        self.ptime_coverage_planning: float = None
        self.ptime_sequencing: float = None
        self.ptime_pipeline: float = None

        #
        self.seg_avg_area: float = None
        self.seg_avg_perimeter: float = None
        self.seg_avg_free_perimeter: float = None
        self.seg_avg_compactness_a: float = None
        self.seg_avg_compactness_b: float = None

        #
        self.l_path_full: float = None
        self.l_path_coverage: float = None
        self.path_yaw_full: float = None
        self.path_yaw_coverage: float = None

        #
        self.area_cleaning_m: float = None
        self.area_cleaned_m: float = None
        self.area_uncleaned_m: float = None
        self.area_overlaped_m: float = None
        self.area_coverage_pc: float = None
        self.area_overlap_pc: float = None

    def get_data_dict(self) -> dict:
        """Convert statistics to dictionary format for serialization."""
        return {
            "map_name": self.map_name,
            "method": self.method,
            "n_zones": int(self.n_zones),
            "n_cells": int(self.n_cells),
            "n_cov_cells": int(self.n_cov_cells),
            "trav_time_full": round(float(self.trav_time_full), 2),
            "trav_time_coverage": round(float(self.trav_time_coverage), 2),
            "ptime_segmentation": round(float(self.ptime_segmentation), 2),
            "ptime_coverage_planning": round(float(self.ptime_coverage_planning), 2),
            "ptime_sequencing": round(float(self.ptime_sequencing), 2),
            "ptime_pipeline": round(float(self.ptime_pipeline), 2),
            "seg_avg_area": round(float(self.seg_avg_area), 2),
            "seg_avg_perimeter": round(float(self.seg_avg_perimeter), 2),
            "seg_avg_free_perimeter": round(float(self.seg_avg_free_perimeter), 2),
            "seg_avg_compactness_a": round(float(self.seg_avg_compactness_a), 2),
            "seg_avg_compactness_b": round(float(self.seg_avg_compactness_b), 2),
            "l_path_full": round(float(self.l_path_full), 2),
            "l_path_coverage": round(float(self.l_path_coverage), 2),
            "path_yaw_full": round(float(self.path_yaw_full), 2),
            "path_yaw_coverage": round(float(self.path_yaw_coverage), 2),
            "area_cleaning_m": round(float(self.area_cleaning_m), 2),
            "area_cleaned_m": round(float(self.area_cleaned_m), 2),
            "area_uncleaned_m": round(float(self.area_uncleaned_m), 2),
            "area_overlaped_m": round(float(self.area_overlaped_m), 2),
            "area_coverage_pc": round(float(self.area_coverage_pc), 2),
            "area_overlap_pc": round(float(self.area_overlap_pc), 2)
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
        self.stats.n_cells = sum(zs.n_cells for zs in zone_stats)
        self.stats.n_cov_cells = sum(zs.n_cov_cells for zs in zone_stats)
        self.stats.trav_time_full = sum(zs.trav_time_full for zs in zone_stats)
        self.stats.trav_time_coverage = sum(zs.trav_time_coverage for zs in zone_stats)
        self.stats.ptime_segmentation = sum(zs.ptime_segmentation for zs in zone_stats)
        self.stats.ptime_coverage_planning = sum(zs.ptime_coverage_planning for zs in zone_stats)
        self.stats.ptime_sequencing = sum(zs.ptime_sequencing for zs in zone_stats)
        self.stats.ptime_pipeline = sum(zs.ptime_pipeline for zs in zone_stats)
        self.stats.seg_avg_area = sum(zs.seg_avg_area for zs in zone_stats) / self.stats.n_zones
        self.stats.seg_avg_perimeter = sum(zs.seg_avg_perimeter for zs in zone_stats) / self.stats.n_zones
        self.stats.seg_avg_free_perimeter = sum(zs.seg_avg_free_perimeter for zs in zone_stats) / self.stats.n_zones
        self.stats.seg_avg_compactness_a = sum(zs.seg_avg_compactness_a for zs in zone_stats) / self.stats.n_zones
        self.stats.seg_avg_compactness_b = sum(zs.seg_avg_compactness_b for zs in zone_stats) / self.stats.n_zones
        self.stats.l_path_full = sum(zs.path_full_length for zs in zone_stats)
        self.stats.l_path_coverage = sum(zs.path_coverage_length for zs in zone_stats)
        self.stats.path_yaw_full = sum(zs.path_yaw_full for zs in zone_stats)
        self.stats.path_yaw_coverage = sum(zs.path_yaw_coverage for zs in zone_stats)
        self.stats.area_cleaning_m = sum(zs.area_cleaning_m for zs in zone_stats)
        self.stats.area_cleaned_m = sum(zs.area_cleaned_m for zs in zone_stats)
        self.stats.area_uncleaned_m = sum(zs.area_uncleaned_m for zs in zone_stats)
        self.stats.area_overlaped_m = sum(zs.area_overlaped_m for zs in zone_stats)
        self.stats.area_coverage_pc = self.stats.area_cleaned_m / self.stats.area_cleaning_m * 100
        self.stats.area_overlap_pc = self.stats.area_overlaped_m / self.stats.area_cleaning_m * 100

        # save result
        self.save_result()

    def save_result(self):
        """ save district statistics to json file. """
        results = self.stats.get_data_dict()
        with open(self.result_file, 'wt', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
