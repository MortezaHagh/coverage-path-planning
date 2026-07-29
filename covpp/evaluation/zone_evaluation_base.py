"""Zone evaluation base classes.

Defines base data structures and interface for computing zone-level KPIs
from generated coverage and sequence paths.
"""
import json
from covpp.base.logger import MyLogger
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.sequence_handler import SequencePaths


class ZoneStatsBase:
    """Lightweight DTO that captures zone-level KPIs.

    Result:
        Instances accumulate coverage/sequence/evaluation metrics that can be
        serialized via :meth:`get_data_dict` and consumed by district aggregators.

    Attributes mirror the JSON payload (map identifiers, timing, path metrics,
    coverage ratios, and areas in px/m). All values default to ``None`` or ``0``
    so evaluators can fill them progressively.
    """

    def __init__(self):
        self.map_name: str = None
        self.method: str = None
        self.ind: int = None  # zone index
        self.n_cells: int = None  # number of cells in zone
        self.n_cov_cells: int = None  # number of cells with coverage paths
        self.trav_time_full: float = 0.0
        self.trav_time_coverage: float = 0.0
        # Processing times for each stage (segmentation, coverage planning, sequencing, total)
        self.ptime_segmentation: float = None
        self.ptime_coverage_planning: float = None
        self.ptime_sequencing: float = None
        self.ptime_pipeline: float = None
        #
        self.full_length = None
        self.coverage_length = None
        self.yaw_path_full = None
        self.yaw_path_coverage = None
        #
        self.cleaning_area_px = None
        self.area_cleaning_m = None
        self.cleaned_area_px = 0
        self.area_cleaned_m = 0
        self.area_uncleaned_m = None
        self.area_overlaped_m = None
        self.area_coverage_pc = None
        self.area_overlap_pc = None

    def get_data_dict(self) -> dict:
        """Serialize statistics to a plain dictionary for reporting.

        Returns:
        - dict: JSON-serializable payload of KPIs and metadata.
        """
        return {
            "map_name": self.map_name,
            "method": self.method,
            "zone_ind": self.ind,
            "n_cells": self.n_cells,
            "n_cov_cells": self.n_cov_cells,
            "trav_time_full": round(float(self.trav_time_full), 2),
            "trav_time_coverage": round(float(self.trav_time_coverage), 2),
            "ptime_segmentation": round(float(self.ptime_segmentation), 2),
            "ptime_coverage_planning": round(float(self.ptime_coverage_planning), 2),
            "ptime_sequencing": round(float(self.ptime_sequencing), 2),
            "ptime_pipeline": round(float(self.ptime_pipeline), 2),
            "full_length": round(float(self.full_length), 2),
            "coverage_length": round(float(self.coverage_length), 2),
            "yaw_path_full": round(float(self.yaw_path_full), 2),
            "yaw_path_coverage": round(float(self.yaw_path_coverage), 2),
            "cleaning_area_px": int(self.cleaning_area_px),
            "area_cleaning_m": round(float(self.area_cleaning_m), 2),
            "cleaned_area_px": int(self.cleaned_area_px),
            "area_cleaned_m": round(float(self.area_cleaned_m), 2),
            "area_uncleaned_m": round(float(self.area_uncleaned_m), 2),
            "area_overlaped_m": round(float(self.area_overlaped_m), 2),
            "area_coverage_pc": round(float(self.area_coverage_pc), 2),
            "area_overlap_pc": round(float(self.area_overlap_pc), 2)
        }


class ZoneEvaluationBase:
    """Base interface for computing zone statistics from generated paths.

    Steps:
        1. Hold references to ``ZoneMapData`` plus the ``SequencePaths`` bundle.
        2. Implement :meth:`evaluate` to fill ``self.stats`` with KPI values.
        3. Optionally persist results via :meth:`save_result`.

    Result:
        ``self.stats`` must be a populated ``ZoneStatsBase`` instance after
        ``evaluate`` completes successfully.
    """

    def __init__(self, zmd: ZoneMapData, paths: SequencePaths):
        self.lg = MyLogger("ZoneStatsBase")
        # attributes
        self.zmd: ZoneMapData = zmd
        self.paths: SequencePaths = paths
        # statistics
        self.stats: ZoneStatsBase = None

    def evaluate(self, times: float = 0.0):
        """Compute zone statistics. Must be implemented by subclasses.

        Args:
        - times: Optional timing info or dict (depends on subclass needs).
        """
        raise NotImplementedError("Zone evaluation subclasses must implement evaluate() method")

    def save_result(self):
        """Persist zone statistics to the zone’s result JSON file. """
        zone_stats = self.stats.get_data_dict()
        with open(self.zmd.result_file, 'wt', encoding='utf-8') as f:
            json.dump(zone_stats, f, indent=4)

    def validate_data(self, times: dict) -> bool:
        """Validate prerequisites before evaluating statistics.

        Checks presence of timing info and required paths; prepares a cleaned map
        fallback when full path is missing.

        Returns:
        - bool: True when safe to compute metrics; False otherwise.
        """
        is_valid = True

        # check times
        if times is None:
            self.lg.error("times dictionary is None. Cannot calculate time statistics.")
            is_valid = False

        # paths: from sequence paths (full path in pixels)
        if self.paths.f_path_px is None or len(self.paths.paths_m) == 0:
            self.lg.error("full path is None or empty. Cannot add cleaned area to map.")
            self.zmd.map_cleaned = self.zmd.map_cleaning.copy()
            is_valid = False

        return is_valid
