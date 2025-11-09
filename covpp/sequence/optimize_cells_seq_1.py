"""Cell sequencing optimization using various algorithms."""
import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.segmentation.cell_processed import PCell


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
        dcfg = PipeLineConfig()

        # sequence results initialization
        self.tour_idxs: list[int] = []  # indices of cells in the optimized sequence
        self.tour_ids: list[int] = []   # ids of cells in the optimized sequence

        # create optimizer
        optimizer: BaseSeqOpt = None
        if dcfg.sequence_method == "simple":
            optimizer = SimpleSeqOpt(cells)
        elif dcfg.sequence_method == "dfs":
            optimizer = DepthFirstSearch(cells, start_cell_idx)
        elif dcfg.sequence_method == "milp":
            if distances is None:
                raise ValueError("Distances matrix is required for MILPSeq optimization.")
            optimizer = MILPSeq(cells, distances, start_cell_idx)
        else:
            raise ValueError(f"Invalid cell sequence method: {dcfg.sequence_method}")

        # check if the optimizer was successful
        if optimizer.tour_idxs is None or len(optimizer.tour_idxs) == 0:
            raise ValueError("No cells sequence found.")

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


# ==============================================================================
# Sequence optimization methods
# ==============================================================================

class BaseSeqOpt:
    """ Base class for sequence optimization of cells. """

    def __init__(self, cells: list[PCell], start_idx: int = 0):
        self.lg = MyLogger("BaseSeqOpt")

        # start cell
        self.start_idx: int = start_idx  # default start index for the tour
        self.start_id: int = cells[start_idx].id

        # cell id to index mapping
        self.cells: list[PCell] = cells
        self.id_idx: dict[int, int] = {c.id: i for i, c in enumerate(cells)}

        # planning cells
        self.ids = [c.id for c in self.cells if c.do_planning]  # planning cell ids
        self.n = len(self.ids)
        if self.n == 0:
            raise ValueError("No planning cells available for sequencing.")

        # solution
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
        self.tour_ids = self.ids
        self.tour_idxs = [self.id_idx[i] for i in self.tour_ids]
        self.success = True


class DepthFirstSearch(BaseSeqOpt):
    def __init__(self, cells: list[PCell], start_cell_idx: int = 0):
        super().__init__(cells, start_cell_idx)
        self.lg.info("using DepthFirstSearch sequence optimization method.")
        self.visited_ids: dict = {i: False for i in self.ids}
        self.ids_depth: dict = {i: 0 for i in self.ids}
        self.create_tour()
        self.tour_idxs = [self.id_idx[i] for i in self.tour_ids]
        self.success = True

    def create_tour(self):
        # Start DFS from the first cell
        start = self.start_id
        self.dfs(start, 0)

    def dfs(self, c_id: int, level: int):
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


class MILPSeq(BaseSeqOpt):
    """ Linear programming approach for sequence optimization. """

    def __init__(self, cells: list[PCell], distances: np.ndarray, start_cell_idx: int = 0):
        super().__init__(cells, start_cell_idx)
        self.lg.info("using MILPSeq sequence optimization method.")

        # planning cell indices
        p_idxs = [self.id_idx[id] for id in self.ids]
        self.tour = None
        self.distm = np.full((self.n, self.n), np.inf, dtype=float)

        # create distance matrix
        for i, ci in enumerate(p_idxs):
            for j, cj in enumerate(p_idxs):
                if i == j:
                    continue
                self.distm[i, j] = distances[ci, cj]
                self.distm[j, i] = distances[cj, ci]

                # to dont consider return to start in cost
                if j == start_cell_idx:
                    self.distm[i, j] = 0.0
        # create tour
        self.create_tour()

    def create_tour(self):
        self.success = self.solve_tsp_milp()
        if not self.success:
            self.lg.error("No optimal solution found.")
            return

        self.tour_ids = [self.ids[i] for i in self.tour]
        self.tour_idxs = [self.id_idx[i] for i in self.tour_ids]

    def solve_tsp_milp(self) -> bool:

        # Create the MILP solver
        solver = pywraplp.Solver.CreateSolver("SCIP")  # Use "CBC" for open-source

        # Variables
        start_node = self.id_idx[self.start_id]
        n = self.n
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
                if i != j:
                    solver.Add(u[i] - u[j] + (n-1) * x[i, j] <= n-2)

        # Objective: Minimize total distance
        solver.Minimize(solver.Sum([self.distm[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j]))

        # Solve
        status = solver.Solve()

        # solution
        if status == pywraplp.Solver.OPTIMAL:
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
        return True


# class GraphSearch(BaseSeqOpt):
#     def __init__(self, cells: list[PCell]):
#         super().__init__(cells)
#         self.lg.info("using GraphSearch sequence optimization method.")
#         self.id_depths = dict()
#         self.create_graph()

#     def create_graph(self):
#         id_depths = {i: 0 for i in self.ids}
#         visited = []
#         current_id = self.start_id
#         id_depths[current_id] = 0

#         # open: [current_id, depth]
#         open_list = [(current_id, 0)]

#         while len(open_list) > 0:

#             # select min depth node
#             current_id, depth = min(open_list, key=lambda x: x[1])
#             open_list.remove((current_id, depth))
#             visited.append(current_id)

#             # expand neighbors
#             current_idx = self.id_idx[current_id]
#             neighbors = [n for n in self.cells[current_idx].adjacent_ids if n in self.ids and n not in visited]
#             for neigh_id in neighbors:
#                 if id_depths[neigh_id] == 0 or depth + 1 < id_depths[neigh_id]:
#                     open_list.append((neigh_id, depth + 1))
#                     id_depths[neigh_id] = depth + 1
#         self.id_depths = id_depths

#     def create_tour(self):
#         tour_ids = [self.start_id]
#         while self.id_depths:
#             current_id = min(self.id_depths, key=self.id_depths.get)
#             tour_ids.append(current_id)
#             self.id_depths.pop(current_id)
#         self.tour_ids = tour_ids
