import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scripts.base.configs import DistrictConfgs
from scripts.boustrophedon.cell import Cell
from scripts.boustrophedon.search import SearchCells


class OptimCellsSeq:
    def __init__(self, mcells: list[Cell]):
        self.ns = "OptimCellsSeq"
        dcfgs = DistrictConfgs()
        print(f"[{self.ns}]: >>>>> Step: creating optimization cells sequence")

        #
        optimizer = None
        if dcfgs.cell_sequence_method == "simple":
            optimizer = SimpleSeq(mcells)
            return
        elif dcfgs.cell_sequence_method == "milp":
            optimizer = MILPSeq(mcells)
        elif dcfgs.cell_sequence_method == "dfs":
            optimizer = DepthFirstSearch(mcells)
        else:
            print(f"[{self.ns}][ERROR]: Invalid cell sequence method: {dcfgs.cell_sequence_method}")
            return

        #
        self.tour = optimizer.tour
        self.set_cells_optim_ind(mcells)
        # self.plot()

    def set_cells_optim_ind(self, mcells: list[Cell]):
        nt = len(self.tour)-1
        for c in mcells[:-1]:
            ind = self.tour.index(c.ind)
            next_cell_ind = None
            if ind < nt:
                next_cell_ind = self.tour[ind + 1]
            c.set_optim_ind(ind, next_cell_ind)

    def plot(self):
        _, ax = plt.subplots(1, 1)
        ax.set_title("Optimized Cells Sequence")
        ax.set_aspect('equal', adjustable='box')

        # #
        # print(self.tour)
        for cell in self.cells:
            ax.plot(cell.cxy[0], cell.cxy[1], "ro")
            text = f"{cell.ind}-{cell.opt_ind}"
            ax.text(cell.cxy[0], cell.cxy[1], text, fontsize=12, color='blue')

        # Draw the tour
        for i in range(len(self.tour)-1):
            x1, y1 = self.cells[self.tour[i]].cxy
            x2, y2 = self.cells[self.tour[i + 1]].cxy
            ax.plot([x1, x2], [y1, y2], 'b-')
        plt.show()


class SimpleSeq:
    def __init__(self, mcells: list[Cell]):
        self.ns = "SimpleSeq"
        print(f"[{self.ns}]: >>>>> Step: creating cells sequence.")
        self.tour = list(range(len(mcells)))


class MILPSeq:
    def __init__(self, mcells: list[Cell]):
        self.ns = "MILPSeq"
        print(f"[{self.ns}]: >>>>> Step: creating cells sequence.")
        self.cells = mcells
        self.n_cells = len(mcells)
        self.search = SearchCells(self.cells)
        self.tour: list[int] = []
        self.distm: np.ndarray = None
        #
        self.create_dist_matrix()
        success = self.solve_tsp_milp()
        if not success:
            self.tour = []
            print(f"[{self.ns}][ERROR]: No optimal solution found.")

    def create_dist_matrix(self):
        n = len(self.cells)
        distm = np.ones((n, n))*np.inf
        for i in range(n):
            for j in range(i+1, n):
                distm[i, j] = self.search.get_distance(i, j)
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

        # Objective: Minimize total distance (no return to start)
        solver.Minimize(
            sum(self.distm[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j)
        )

        # Constraints:
        # 1. Each city entered at most once (except start)
        for j in range(1, n):
            solver.Add(sum(x[i, j] for i in range(n) if i != j) == 1)
        # 2. Each city exited at most once (except end)
        for i in range(n - 1):
            solver.Add(sum(x[i, j] for j in range(n) if j != i) == 1)
        # 3. Start node (0) has in-degree 0, out-degree 1
        solver.Add(sum(x[i, 0] for i in range(n) if i != 0) == 0)
        solver.Add(sum(x[0, j] for j in range(n) if j != 0) == 1)
        # 4. End node (n-1) has out-degree 0, in-degree 1
        solver.Add(sum(x[n-1, j] for j in range(n) if j != n-1) == 0)
        solver.Add(sum(x[i, n-1] for i in range(n) if i != n-1) == 1)

        # 5. MTZ subtour elimination (for all except start)
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    solver.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

        # Solve
        status = solver.Solve()

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
            self.tour = tour

        else:
            print("[solve_tsp_milp]: No optimal solution found.")
            return False
        #
        return True


class DepthFirstSearch:
    def __init__(self, cells: list[Cell]):
        self.ns = "DepthFirstSearch"
        print(f"[{self.ns}]: >>>>> Step: creating depth first search cells sequence")
        #
        self.cells = cells
        self.n_cells = len(cells)
        self.tour: list[int] = []
        self.visited: list[bool] = [False] * self.n_cells
        self.create_tour()

    def create_tour(self):
        # Start DFS from the first cell
        self.dfs(0)
        # for i, cell in enumerate(self.cells):
        #     if not self.visited[i]:
        #         self.dfs(cell.ind)

    def dfs(self, current: int):
        self.visited[current] = True
        self.tour.append(current)
        neighbors = self.cells[current].adjacent_inds
        for neighbor in neighbors:
            if not self.visited[neighbor]:
                self.dfs(neighbor)
