""" Linear programming approach for sequence optimization. """
import numpy as np
from ortools.linear_solver import pywraplp
from covpp.sequence.sequence_optimizer_base import SeqOptimizerBase


class SeqOptimizer(SeqOptimizerBase):
    """MILP-based TSP solver for sequencing coverage cells.

    Steps:
        1. Build a reduced distance matrix that only considers planning cells.
        2. Formulate a single-tour TSP using OR-Tools' MILP backend.
        3. Decode solver output into ``tour_ids``/``tour_idxs``.

    Result:
        Ordered visit sequence that minimizes travel distance according to the
        provided pairwise distance matrix.
    """

    def __init__(self):
        super().__init__()
        self.lg.info("using MilpSO sequence optimization method.")
        self.distm: np.ndarray = None  # distance matrix for planning cells
        self.tour: list[int] = None

    def create_tour(self):
        """Assemble MILP inputs, solve the TSP, and translate the solution."""
        # planning cell indices
        p_idxs = [self.id_idx[id] for id in self.ids]
        self.tour = None
        self.distm = np.full((self.n, self.n), np.inf, dtype=float)

        # create distance matrix
        for i, ci in enumerate(p_idxs):
            for j, cj in enumerate(p_idxs):
                if i == j:
                    continue
                self.distm[i, j] = self.distances[ci, cj]
                self.distm[j, i] = self.distances[cj, ci]

                # to dont consider return to start in cost
                if j == self.start_idx:
                    self.distm[i, j] = 0.0

        # solve TSP using MILP
        self.success = self.solve_tsp_milp()
        if not self.success:
            self.lg.error("No optimal solution found.")
            return

        self.tour_ids = [self.ids[i] for i in self.tour]
        self.tour_idxs = [self.id_idx[i] for i in self.tour_ids]

    def solve_tsp_milp(self) -> bool:
        """Solve the reduced TSP via OR-Tools and update ``self.tour`` if optimal."""

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


# class GraphSearch(SeqOptimizerBase):
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
