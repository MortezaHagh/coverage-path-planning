"""Cell sequencing optimization base class."""
import numpy as np
import matplotlib.pyplot as plt
from covpp.base.logger import MyLogger
from covpp.segmentation.cell_planning import PCell


class SeqOptimizerBase:
    """Reusable scaffolding for sequencing plug-ins.

    Steps:
        1. ``run`` stores the planning cells plus any optional distance matrix.
        2. ``create_tour`` (implemented by subclasses) computes the visiting order.
        3. ``set_cells_optim_ind`` writes the sequence metadata back into each cell.

    Result:
        Subclasses must populate ``tour_idxs`` and ``tour_ids`` to describe the visit
        order. ``success`` indicates whether the optimizer produced a feasible tour.
    """

    def __init__(self):
        self.lg = MyLogger("SeqOptimizerBase")

        # attributes
        self.cells: list[PCell] = []          # list of cells to be sequenced
        self.distances: np.ndarray = None     # distance matrix between cells
        self.start_idx: int = 0               # default start index for the tour
        self.start_id: int = -1               # default start id for the tour
        self.ids: list[int] = []              # planning cell ids
        self.id_idx: dict[int, int] = dict()  # cell id to index mapping
        self.n: int = 0                       # number of planning cells

        # solution
        self.tour_idxs: list[int] = []
        self.tour_ids: list[int] = []
        self.success = True

    def run(self, cells: list[PCell], start_idx: int = 0, distances: np.ndarray = None):
        """Bootstrap optimizer state and delegate to ``create_tour``.

        Args:
            cells: Planning cells that contain adjacency, paths, and metadata.
            start_idx: Index of the seed cell in ``cells``.
            distances: Optional matrix of pairwise costs (meters or pixels).

        Result:
            Sets ``tour_idxs``/``tour_ids`` and annotates every cell with
            ``opt_id``/``next_cell_id`` so downstream handlers can connect paths.
        """

        self.cells = cells
        self.distances = distances
        self.start_idx = start_idx
        self.start_id = cells[start_idx].id
        self.id_idx = {c.id: i for i, c in enumerate(cells)}
        self.ids = [c.id for c in self.cells if c.do_planning]
        self.n = len(self.ids)
        if self.n == 0:
            raise ValueError("No planning cells available for sequencing.")

        # create tour
        self.create_tour()
        self.set_cells_optim_ind()

    def create_tour(self):
        """Create ``tour_idxs`` and ``tour_ids`` for the sequencing strategy."""
        raise NotImplementedError("This method should be overridden in subclasses.")

    def set_cells_optim_ind(self):
        """Assign optimized sequence indices and next cell references to cells."""
        for i, idx in enumerate(self.tour_idxs[:-1]):
            next_idx = self.tour_idxs[i + 1]
            self.cells[idx].set_optim_id(idx, next_idx)

    def plot(self):
        """ Plot the optimized cell sequence. """
        _, ax = plt.subplots(1, 1)
        ax.set_title("Optimized Cells Sequence")
        ax.set_aspect('equal', adjustable='box')

        for cell in self.cells:
            ax.plot(cell.cxy[0], cell.cxy[1], "ro")
            text = f"{cell.id}-{cell.opt_id}"
            ax.text(cell.cxy[0], cell.cxy[1], text, fontsize=12, color='blue')

        # Draw the tour
        for i in range(len(self.tour_idxs)-1):
            x1, y1 = self.cells[self.tour_idxs[i]].cxy
            x2, y2 = self.cells[self.tour_idxs[i + 1]].cxy
            ax.plot([x1, x2], [y1, y2], 'b-')
        plt.show()
