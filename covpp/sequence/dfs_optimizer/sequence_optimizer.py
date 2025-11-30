""" Depth-First Search based sequence optimization method. """
from covpp.sequence.sequence_optimizer_base import SeqOptimizerBase


class SeqOptimizer(SeqOptimizerBase):
    """ Depth-First Search based sequence optimization method. """

    def __init__(self):
        super().__init__()
        self.lg.update_name("DfsSeqOptimizer")
        self.lg.info("using DfsSeqOptimizer sequence optimization method.")
        self.visited_ids = {}
        self.ids_depth = {}

    def create_tour(self):
        self.visited_ids = {i: False for i in self.ids}
        self.ids_depth = {i: 0 for i in self.ids}
        start = self.start_id
        self.dfs(start, 0)
        self.tour_idxs = [self.id_idx[i] for i in self.tour_ids]

    def dfs(self, c_id: int, level: int):
        """ Depth-First Search, recursive. """
        self.visited_ids[c_id] = True
        self.tour_ids.append(c_id)
        idx = self.id_idx[c_id]
        # Get adjacent planning cells
        neighbors = [n for n in self.cells[idx].adjacent_ids if n in self.ids]
        levels = []
        for neigh_id in neighbors:
            if not self.visited_ids[neigh_id]:
                level = self.dfs(neigh_id, 0)
                levels.append(level)
        new_level = max(levels) if levels else 0
        self.ids_depth[c_id] = new_level
        return new_level + 1
