""" Simple sequence optimization method that returns cells in given order. """
from covpp.sequence.sequence_optimizer_base import SeqOptimizerBase


class SeqOptimizer(SeqOptimizerBase):
    """Pass-through sequencing strategy.

    Steps:
        1. Keep the cell order as received from segmentation.
        2. Mirror that order into ``tour_ids``/``tour_idxs``.

    Result:
        Sequencing metadata matches the original list, useful for debugging.
    """

    def __init__(self):
        super().__init__()
        self.lg.update_name("SimpleSeqOptimizer")
        self.lg.info("using SimpleSeqOptimizer sequence optimization method.")

    def create_tour(self):
        """Copy IDs/indexes verbatim so downstream stages preserve ordering."""
        self.tour_ids = self.ids
        self.tour_idxs = [self.id_idx[i] for i in self.tour_ids]
