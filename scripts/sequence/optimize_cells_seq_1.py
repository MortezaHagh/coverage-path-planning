"""Cell sequencing optimization using various algorithms."""
import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scripts.base.logger import MyLogger
from scripts.base.configs import PipeLineConfig
from scripts.segmentation.cell_processed import PCell


class OptimizeCellsSeq:
    """Optimize cell visit order using configurable sequencing algorithms.

    Supports multiple optimization strategies for determining the optimal
    order to visit coverage cells, minimizing transition distances and time.

    Available Methods:
        - simple: Use cells in original order (no optimization)
        - dfs: Depth-first search for connected cell traversal
        - milp: Mixed Integer Linear Programming for optimal TSP solution
    """

    def __init__(self, cells: list[PCell], distances: np.ndarray = None, start_cell_idx: int = 0):
        self.lg = MyLogger("OptimizeCellsSeq")
        self.lg.info1("Step: creating optimized cell sequence")

        # initialize
        self.cells: list[PCell] = cells
        dcfgs = PipeLineConfig()

        # sequence results initialization
        self.tour_idxs: list[int] = []  # indices of cells in the optimized sequence
        self.tour_ids: list[int] = []   # ids of cells in the optimized sequence

        # create optimizer
        optimizer: BaseSeqOpt = None
        if dcfgs.sequence_method == "simple":
            optimizer = SimpleSeqOpt(cells)
        elif dcfgs.sequence_method == "dfs":
            optimizer = DepthFirstSearch(cells, start_cell_idx)
        elif dcfgs.sequence_method == "milp":
            # cell graph
            if distances is None:
                raise ValueError("Distances matrix is required for MILPSeq optimization.")
            optimizer = MILPSeq(cells, distances, start_cell_idx)
        else:
            self.lg.error(f"Invalid cell sequence method: {dcfgs.sequence_method}")
            return

        # check if the optimizer was successful
        if optimizer.tour_idxs is None or len(optimizer.tour_idxs) == 0:
            self.lg.error("No cells sequence found.")
            return

        # save the tour
        self.tour_idxs = optimizer.tour_idxs
        self.tour_ids = optimizer.tour_ids
        self.set_cells_optim_ind()
        # self.plot()

    def set_cells_optim_ind(self):
        """Assign optimized sequence indices and next cell references to cells."""
        for i, idx in enumerate(self.tour_idxs[:-1]):
            next_idx = self.tour_idxs[i + 1]
            self.cells[idx].set_optim_id(idx, next_idx)

    def plot(self):
        _, ax = plt.subplots(1, 1)
        ax.set_title("Optimized Cells Sequence")
        ax.set_aspect('equal', adjustable='box')

        for cell in self.cells:
            ax.plot(cell.cxy[0], cell.cxy[1], "ro")
            text = f"{cell.id}-{cell.opt_id}"
            ax.text(cell.cxy[0], cell.cxy[1], text, fontsize=12, color='blue')

        # Draw the tour
        for i in range(len(self.tour)-1):
            x1, y1 = self.cells[self.tour[i]].cxy
            x2, y2 = self.cells[self.tour[i + 1]].cxy
            ax.plot([x1, x2], [y1, y2], 'b-')
        plt.show()


class BaseSeqOpt:
    """ Base class for sequence optimization of cells. """

    def __init__(self, cells: list[PCell], start_idx: int = 0):
        self.lg = MyLogger("BaseSeqOpt")
        self.cells: list[PCell] = cells
        self.c_id_idx_dict: dict[int, int] = {c.id: i for i, c in enumerate(cells)}  # cell id to index mapping
        self.start_idx: int = start_idx  # default start index for the tour
        self.tour_idxs: list[int] = []
        self.tour_ids: list[int] = []
        self.success = False

    def create_tour(self):
        raise NotImplementedError("This method should be overridden in subclasses.")


class SimpleSeqOpt(BaseSeqOpt):
    def __init__(self, cells: list[PCell]):
        super().__init__(cells)
        self.lg.info("using SimpleSeqOpt sequence optimization method.")

    def create_tour(self):
        self.tour_idxs = list(range(len(self.cells)))
        self.tour_ids = [c.id for c in self.cells]
        self.success = True


class DepthFirstSearch(BaseSeqOpt):
    def __init__(self, cells: list[PCell], start_cell_idx: int = 0):
        super().__init__(cells, start_cell_idx)
        self.lg.info("using DepthFirstSearch sequence optimization method.")
        #
        self.c_ids = [c.id for c in cells]
        self.visited_ids_dict: dict = {i: False for i in self.c_ids}
        self.depth_ids_dict: dict = {i: 0 for i in self.c_ids}
        #
        self.create_tour()
        self.tour_idxs = [self.c_id_idx_dict[i] for i in self.tour_ids]

    def create_tour(self):
        # Start DFS from the first cell
        start = self.c_ids[0]
        self.dfs(start, 0)
        # for i, cell in enumerate(self.cells):
        #     if not self.visited[i]:
        #         self.dfs(cell.id)
        self.success = True

    def dfs(self, c_id: int, level: int):
        self.visited_ids_dict[c_id] = True
        self.tour_ids.append(c_id)
        idx = self.c_id_idx_dict[c_id]
        neighbors = self.cells[idx].adjacent_ids
        neighbors = [n for n in neighbors if n in self.c_ids]
        new_level = 0
        for neigh_id in neighbors:
            if not self.visited_ids_dict[neigh_id]:
                level = self.dfs(neigh_id, new_level)
                new_level = max(level, new_level)
        self.depth_ids_dict[c_id] = new_level
        return new_level + 1


class MILPSeq(BaseSeqOpt):
    def __init__(self, cells: list[PCell], distances: np.ndarray, start_cell_idx: int = 0):
        super().__init__(cells, start_cell_idx)
        self.lg.info("using MILPSeq sequence optimization method.")

        # planning cells
        p_cells = [c for c in cells if c.do_planning]
        p_idxs = [i for i, c in enumerate(cells) if c.do_planning]
        self.p_cells = p_cells
        self.n_p_cells = len(p_idxs)

        #
        if self.n_p_cells == 0:
            self.lg.error("No planning cells found. Cannot create tour.")
            return

        #
        self.tour = None
        self.distm = np.full((self.n_p_cells, self.n_p_cells), np.inf, dtype=float)

        # create distance matrix
        for i, ci in enumerate(p_idxs):
            for j, cj in enumerate(p_idxs):
                if i == j:
                    continue
                self.distm[i, j] = distances[ci, cj]
                self.distm[j, i] = distances[cj, ci]

        #
        self.create_tour()

    def create_tour(self):
        self.success = self.solve_tsp_milp()
        if not self.success:
            self.lg.error("No optimal solution found.")
            return

        c_ids = [c.id for c in self.p_cells]
        self.tour_ids = [c_ids[i] for i in self.tour]
        self.tour_idxs = [self.c_id_idx_dict[i] for i in self.tour_ids]

    def solve_tsp_milp(self) -> bool:
        solver = pywraplp.Solver.CreateSolver("SCIP")  # Use "CBC" for open-source

        #
        start_node = self.start_idx
        n = self.n_p_cells
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = solver.BoolVar(f"x_{i}_{j}")

        # Constraints:
        # 1. Each city entered at most once (except start)
        for j in range(n):
            solver.Add(solver.Sum([x[i, j] for i in range(n) if i != j]) == 1)
        # 2. Each city exited at most once (except end)
        for i in range(n):
            solver.Add(solver.Sum([x[i, j] for j in range(n) if j != i]) == 1)

        # 5. MTZ subtour elimination (for all except start)
        u = [solver.NumVar(0, n-1, f"u_{i}") for i in range(n)]
        for i in range(1, n):
            for j in range(1, n):
                if i != j and i != start_node and j != start_node:
                    solver.Add(u[i] - u[j] + (n-1) * x[i, j] <= n-2)

        # Objective: Minimize total distance
        solver.Minimize(
            solver.Sum([self.distm[i, j] * x[i, j] for i in range(n) for j in range(n) if (i != j and j != start_node)])
        )

        # Solve
        status = solver.Solve()

        # solution
        if status == pywraplp.Solver.OPTIMAL:
            # self.lg.info("Optimal solution found!")
            # self.lg.info(f"Total cost = {solver.Objective().Value()}\n")

            # Extract the path (not a cycle)
            tour = [start_node]
            current = start_node
            while len(tour) < n:
                for j in range(n):
                    if current != j and x[current, j].solution_value() > 0.5:
                        tour.append(j)
                        current = j
                        break
            self.tour = tour
        else:
            self.lg.error("[solve_tsp_milp]: No optimal solution found.")
            return False
        #
        return True
