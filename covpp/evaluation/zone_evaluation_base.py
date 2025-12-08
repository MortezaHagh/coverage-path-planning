""" Zone evaluation base classes."""
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
        """ evaluate zone statistics. To be implemented in derived classes. """
        raise NotImplementedError("Zone evaluation subclasses must implement evaluate() method")

    def save_result(self):
        """ save zone statistics and planner statistics of the zone to json file. """
        zone_stats = self.stats.get_data_dict()
        with open(self.zmd.result_file, 'wt', encoding='utf-8') as f:
            json.dump(zone_stats, f, indent=4)

    def validate_data(self, times: dict) -> bool:
        """ check if data are valid for evaluation. """
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
