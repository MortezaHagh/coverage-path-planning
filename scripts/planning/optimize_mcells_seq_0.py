""" using CellsGraph to create optimized sequence of cells to be covered. """

import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfigs
from scripts.planning.cells_graph_0 import CellsGraph
from scripts.planning.boustrophedon.cell import Cell


class OptimCellsSeq:
    def __init__(self, mcells: list[Cell]):
        self.ns = "OptimCellsSeq"
        self.lg = MyLogger(self.ns)
        dcfgs = DistrictConfigs()
        self.lg.info1("Step: creating optimized cell sequence")

        # cell graph
        self.cells_graph = CellsGraph(mcells)

        #
        self.tour_idxs = None
        self.tour_ids = None
        self.cells_seq_length = None

        # filter based on do_planning
        pmcells = [c for c in mcells if c.do_planning]

        #
        optimizer = None
        if dcfgs.cell_sequence_method == "s1_simple":
            optimizer = SimpleSeqOpt(pmcells)
        elif dcfgs.cell_sequence_method == "s2_dfs":
            optimizer = DepthFirstSearch(pmcells)
        elif dcfgs.cell_sequence_method == "s3_milp":
            optimizer = MILPSeq(self.cells_graph, mcells)
        else:
            self.lg.error(f"Invalid cell sequence method: {dcfgs.cell_sequence_method}")
            return

        #
        if optimizer.tour_idxs is None or len(optimizer.tour_idxs) == 0:
            self.lg.error("No cells sequence found.")
            return
        #
        self.tour_idxs = optimizer.tour_idxs
        self.tour_ids = optimizer.tour_ids
        self.set_cells_optim_ind(pmcells)
        self.cal_cells_seq_length()
        # self.plot()

    def set_cells_optim_ind(self, mcells: list[Cell]):
        new_cells = [mcells[i] for i in self.tour_idxs]
        for i, c in enumerate(new_cells):
            ind = i
            next_cell_id = None
            if i < len(new_cells) - 1:
                next_cell_id = new_cells[i + 1].id
            c.set_optim_id(ind, next_cell_id)

    def cal_cells_seq_length(self) -> float:
        if len(self.tour_ids) == 1:
            self.cells_seq_length = 0.0
            return
        total_length = 0.0
        for i in range(len(self.tour_ids) - 1):
            i1 = self.tour_ids[i]
            i2 = self.tour_ids[i+1]
            d = self.cells_graph.get_distance(i1, i2)
            if d is None:
                self.lg.error(f"Distance between cells {i1} and {i2} is None.")
                continue
            total_length += d
        self.cells_seq_length = total_length

    def plot(self):
        _, ax = plt.subplots(1, 1)
        ax.set_title("Optimized Cells Sequence")
        ax.set_aspect('equal', adjustable='box')

        # #
        # self.lg.info(self.tour)
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


class SimpleSeqOpt:
    def __init__(self, mcells: list[Cell]):
        self.ns = "SimpleSeqOpt"
        print(f"[{self.ns}]: creating cells sequence.")
        self.tour_idxs = list(range(len(mcells)))
        self.tour_ids = [c.id for c in mcells]


class DepthFirstSearch:
    def __init__(self, cells: list[Cell]):
        self.ns = "DepthFirstSearch"
        print(f"[{self.ns}]: creating cells sequence")
        #
        self.cells = cells
        self.c_ids = [c.id for c in cells]
        self.c_id_idx_dict = {c.id: i for i, c in enumerate(cells)}
        self.tour_ids: list[int] = []
        self.visited_ids_d: dict = {i: False for i in self.c_ids}
        self.depth_ids_d: dict = {i: 0 for i in self.c_ids}
        self.create_tour()
        self.tour_idxs = [self.c_id_idx_dict[i] for i in self.tour_ids]

    def create_tour(self):
        # Start DFS from the first cell
        start = self.c_ids[0]
        self.dfs(start, 0)
        # for i, cell in enumerate(self.cells):
        #     if not self.visited[i]:
        #         self.dfs(cell.id)

    def dfs(self, c_id: int, level: int):
        self.visited_ids_d[c_id] = True
        self.tour_ids.append(c_id)
        ind = self.c_id_idx_dict[c_id]
        neighbors = self.cells[ind].adjacent_ids
        neighbors = [n for n in neighbors if n in self.c_ids]
        new_level = 0
        for neigh_id in neighbors:
            if not self.visited_ids_d[neigh_id]:
                level = self.dfs(neigh_id, new_level)
                new_level = max(level, new_level)
        self.depth_ids_d[c_id] = new_level
        return new_level + 1


class MILPSeq:
    def __init__(self, search: CellsGraph, mcells: list[Cell]):
        self.ns = "MILPSeq"
        print(f"[{self.ns}]: creating cells sequence.")

        pmcells = [c for c in mcells if c.do_planning]
        self.cells = pmcells
        self.n_cells = len(pmcells)
        self.ids = [c.id for c in self.cells]

        #
        self.tour_idxs: list[int] = []
        self.cells_graph = search
        self.distm: np.ndarray = None

        #
        self.create_dist_matrix()
        success = self.solve_tsp_milp()
        if not success:
            print(f"[{self.ns}][ERROR]: No optimal solution found.")
        self.tour_ids = [self.ids[i] for i in self.tour_idxs]

    def create_dist_matrix(self):
        n = self.n_cells
        distm = np.ones((n, n))*np.inf
        for i in range(n):
            for j in range(i+1, n):
                distm[i, j] = self.cells_graph.get_distance(self.ids[i], self.ids[j])
                # distm[i, j] = self.cells_graph.get_distance(i, j)
                distm[j, i] = distm[i, j]
        self.distm = distm

    def solve_tsp_milp(self) -> bool:
        solver = pywraplp.Solver.CreateSolver("SCIP")  # Use "CBC" for open-source

        n = self.n_cells
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

        # MTZ variables (u[i] = position of node i in the tour)
        u = [solver.IntVar(1, n, f"u_{i}") for i in range(n)]

        # Objective: Minimize total distance
        solver.Minimize(
            sum(self.distm[i, j] * x[i, j] for i in range(n) for j in range(n) if (i != j and j != 0))
        )

        # Constraints:
        # 1. Each city entered at most once (except start)
        for j in range(n):
            solver.Add(sum(x[i, j] for i in range(n) if i != j) == 1)
        # 2. Each city exited at most once (except end)
        for i in range(n):
            solver.Add(sum(x[i, j] for j in range(n) if j != i) == 1)

        # 5. MTZ subtour elimination (for all except start)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    solver.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

        # Solve
        status = solver.Solve()

        # solution
        if status == pywraplp.Solver.OPTIMAL:
            # print("Optimal solution found!")
            # print(f"Total cost = {solver.Objective().Value()}\n")

            # Extract the path (not a cycle)
            tour = [0]
            while len(tour) < n:
                last = tour[-1]
                found = False
                for j in range(n):
                    if j != last and x[last, j].solution_value() > 0.5:
                        tour.append(j)
                        found = True
                        break
                if not found:
                    break
            self.tour_idxs = tour

        else:
            print("[solve_tsp_milp]: No optimal solution found.")
            return False
        #
        return True
