""" Simple sequence optimization method that returns cells in given order. """
from covpp.sequence.sequence_optimizer_base import SeqOptimizerBase


class SeqOptimizer(SeqOptimizerBase):
    """ Simple sequence optimization method that returns cells in given order. """

    def __init__(self):
        super().__init__()
        self.lg.update_name("SimpleSeqOptimizer")
        self.lg.info("using SimpleSeqOptimizer sequence optimization method.")

    def create_tour(self):
        self.tour_ids = self.ids
        self.tour_idxs = [self.id_idx[i] for i in self.tour_ids]
