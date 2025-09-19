from scripts.base.logger import MyLogger
from scripts.sequence.sequence import Sequence
from scripts.evaluation.zone_evaluation_simple import ZoneEvaluationSimple
from scripts.evaluation.zone_evaluation_base import ZoneStatsBase, ZoneEvaluationBase


class ZoneEvaluationHandler:
    """Zone evaluation handler class to evaluate zone statistics.
    """

    def __init__(self, sequence: Sequence):
        self.lg = MyLogger("ZoneEvaluationHandler")
        self.zmd = sequence.zmd
        self.sequence = sequence
        # statistics
        self.stats: ZoneStatsBase = None
        self.evaluator: ZoneEvaluationBase = None

    def evaluate(self, duration: float = 0.0):
        # evaluate zone statistics based on the evaluator (currently only one evaluator)
        self.evaluator = ZoneEvaluationSimple(self.sequence)
        self.evaluator.evaluate(duration=duration)
        self.stats = self.evaluator.stats
