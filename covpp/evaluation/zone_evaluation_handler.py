"""Zone evaluation handler for computing coverage performance metrics."""
from covpp.district.zone_map_data import ZoneMapData
from covpp.sequence.sequence_handler import SequencePaths
from covpp.evaluation.zone_evaluation_simple import ZoneEvaluationSimple
from covpp.evaluation.zone_evaluation_base import ZoneStatsBase, ZoneEvaluationBase


class ZoneEvaluationHandler:
    """Handler for zone-level performance evaluation and statistics computation."""

    def __init__(self, zmd: ZoneMapData):
        self.zmd = zmd
        # statistics
        self.stats: ZoneStatsBase = None
        self.evaluator: ZoneEvaluationBase = None

    def evaluate(self, paths: SequencePaths, times: dict = None):
        """Compute zone coverage statistics based on generated paths."""
        # evaluate zone statistics based on the evaluator (currently only one evaluator)
        self.evaluator = ZoneEvaluationSimple(self.zmd, paths)
        self.evaluator.evaluate(times=times)
        self.stats = self.evaluator.stats
