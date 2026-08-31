"""Zone evaluation handler for computing coverage performance metrics."""
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.sequence_handler import SequencePaths
from covpp.evaluation.zone_evaluation_mh import ZoneEvaluation
from covpp.evaluation.zone_evaluation_base import ZoneStatsBase, ZoneEvaluationBase
from covpp.segmentation.segmentation_metrics import SegmentationMetrics


class ZoneEvaluationHandler:
    """Handler for zone-level performance evaluation and statistics computation.

    Attributes:
        zmd (ZoneMapData): zone map data object.
        stats (ZoneStatsBase): computed statistics for the zone.
        evaluator (ZoneEvaluationBase): evaluator used for computing statistics.

    Inputs:
        zmd (ZoneMapData): zone map data object.
        paths (SequencePaths): generated coverage paths for the zone.
        times (dict, optional): timing information for various planning stages.
    """

    def __init__(self, zmd: ZoneMapData):
        self.zmd = zmd
        self.stats: ZoneStatsBase = None
        self.evaluator: ZoneEvaluationBase = None

    def evaluate(self, paths: SequencePaths, segmentation_metrics: SegmentationMetrics, times: dict = None):
        """Compute zone coverage statistics based on generated paths."""

        # evaluate zone statistics based on the evaluator (currently only one evaluator)
        self.evaluator = ZoneEvaluation(self.zmd, paths, segmentation_metrics)
        self.evaluator.evaluate(times=times)
        self.stats = self.evaluator.stats

    def get_stats(self) -> ZoneStatsBase:
        """Return the computed zone statistics."""
        return self.stats
