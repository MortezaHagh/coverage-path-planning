import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scripts.decomposition.cell import Cell
from scripts.decomposition.search import SearchCells


class OptimCellsSeq:
    def __init__(self, mcells: list[Cell]):
        self.ns = "OptimCellsSeq"
        print(f"[{self.ns}]: >>>>> Step: creating optimization cells sequence")

        #
        self.cells = mcells
        self.n_cells = len(mcells)
        self.search = SearchCells(self.cells)

        #
        self.distm: np.ndarray = None
        self.tour: list[int] = None

        #
        self.create_dist_matrix()
        self.solve_tsp_milp()

        #
        self.plot()

    def create_dist_matrix(self):
        n = len(self.cells)
        distm = np.ones((n, n))*np.inf
        for i in range(n):
            for j in range(i+1, n):
                distm[i, j] = self.search.get_distance(i, j)
                distm[j, i] = distm[i, j]
        self.distm = distm

    def plot(self):
        _, ax = plt.subplots(1, 1)
        ax.set_title("Optimized Cells Sequence")
        ax.set_aspect('equal', adjustable='box')

        #
        print(" ============ tour")

        print(self.tour)
        print(" ============ ")
        for cell in self.cells:
            ax.plot(cell.cxy[0], cell.cxy[1], "ro")
            ax.text(cell.cxy[0], cell.cxy[1], str(cell.ind), fontsize=12, color='blue')

        # Draw the tour
        if self.tour:
            for i in range(len(self.tour) - 1):
                x1, y1 = self.cells[self.tour[i]].cxy
                x2, y2 = self.cells[self.tour[i + 1]].cxy
                ax.plot([x1, x2], [y1, y2], 'b-')
            # Connect the last point to the first
            x1, y1 = self.cells[self.tour[-1]].cxy
            x2, y2 = self.cells[self.tour[0]].cxy
            ax.plot([x1, x2], [y1, y2], 'b-')

        #
        plt.show()

    def solve_tsp_milp(self):
        solver = pywraplp.Solver.CreateSolver("SCIP")  # Use "CBC" for open-source

        # Variables
        n = self.n_cells
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j] = solver.IntVar(0, 1, f"x_{i}_{j}")

        # MTZ variables (u[i] = position of node i in the tour)
        u = [solver.IntVar(1, n - 1, f"u_{i}") for i in range(n)]

        # Objective: Minimize total distance
        solver.Minimize(sum(self.distm[i, j] * x[i, j] for i in range(n) for j in range(n) if i != j))

        # Constraints:
        # 1. Each city entered exactly once
        for j in range(n):
            solver.Add(sum(x[i, j] for i in range(n) if i != j) == 1)

        # 2. Each city exited exactly once
        for i in range(n):
            solver.Add(sum(x[i, j] for j in range(n) if j != i) == 1)

        # 3. MTZ subtour elimination
        for i in range(1, n):
            for j in range(1, n):
                if i != j:
                    solver.Add(u[i] - u[j] + n * x[i, j] <= n - 1)

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL:
            print("Optimal solution found!")
            print(f"Total cost = {solver.Objective().Value()}\n")

            # Extract the tour
            tour = [0]  # Start at city 0
            while len(tour) < n:
                last = tour[-1]
                for j in range(n):
                    if j != last and x[last, j].solution_value() > 0.5:
                        tour.append(j)
                        break
            tour.append(0)  # Return to start
            self.tour = tour

        else:
            print("No optimal solution found.")
