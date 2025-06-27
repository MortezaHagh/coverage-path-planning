import numpy as np
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfgs
from scripts.planning.base_cell import BaseCell
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.connect_paths import FindPathInMap


class CellDistances:
    """ Calculate distances between cells """
    lg: MyLogger
    zmd: ZoneMapData
    distances: np.ndarray

    def __init__(self, mcells: list, zmd: ZoneMapData):
        self.ns = "CellDistances"
        self.lg = MyLogger(self.ns)
        self.lg.info1("Step: calculate cell distances")

        #
        self.zmd = zmd

        # calculate distances
        # self.distances = self.calculate_direct_distances(mcells)
        self.distances = self.calculate_distances_on_map(mcells)

    def calculate_direct_distances(self, mcells: list[BaseCell]) -> np.ndarray:
        n = len(mcells)
        dist_matrix = np.zeros((n, n), dtype=float)
        # dist_matrix = np.full((n, n), np.inf, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if mcells[i].end_point is None or mcells[j].start_point is None:
                    # self.lg.warn(f"Cell {mcells[i].id} or {mcells[j].id} has no start or end point.")
                    continue

                dist_matrix[i, j] = np.linalg.norm(np.array(mcells[i].end_point.xy) - np.array(mcells[j].start_point.xy))
                dist_matrix[j, i] = np.linalg.norm(np.array(mcells[j].end_point.xy) - np.array(mcells[i].start_point.xy))
        return dist_matrix

    def calculate_distances_on_map(self, mcells: list[BaseCell]) -> np.ndarray:
        n = len(mcells)
        dist_matrix = np.full((n, n), np.inf, dtype=float)

        fp = FindPathInMap(self.zmd)

        n = len(mcells)
        dist_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if mcells[i].end_point is None or mcells[j].start_point is None:
                    # self.lg.warn(f"Cell {mcells[i].id} or {mcells[j].id} has no start or end point.")
                    continue
                dist_matrix[i, j] = fp.distance(mcells[i].end_point, mcells[j].start_point)
                dist_matrix[j, i] = fp.distance(mcells[j].end_point, mcells[i].start_point)
        return dist_matrix


class OptimCellsSeq:
    ns: str
    lg: MyLogger
    mcells: list[BaseCell]
    tour_idxs: list[int]
    tour_ids: list[int]

    def __init__(self, mcells: list[BaseCell], zmd: ZoneMapData):
        self.ns = "OptimCellsSeq"
        self.lg = MyLogger(self.ns)
        self.lg.info1("Step: creating optimized cell sequence")

        #
        self.mcells = mcells
        dcfgs = DistrictConfgs()

        #
        self.tour_idxs: list = None
        self.tour_ids: list = None

        #
        optimizer: BaseSeqOpt = None
        if dcfgs.cell_sequence_method == "s1_simple":
            optimizer = SimpleSeqOpt(mcells)
        elif dcfgs.cell_sequence_method == "s2_dfs":
            optimizer = DepthFirstSearch(mcells)
        elif dcfgs.cell_sequence_method == "s3_milp":
            # cell graph
            distances = CellDistances(mcells, zmd)
            optimizer = MILPSeq(mcells, distances.distances)
        else:
            self.lg.error(f"Invalid cell sequence method: {dcfgs.cell_sequence_method}")
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
        for i, idx in enumerate(self.tour_idxs[:-1]):
            next_idx = self.tour_idxs[i + 1]
            self.mcells[idx].set_optim_id(idx, next_idx)

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
    ns: str
    lg: MyLogger
    mcells: list[BaseCell]
    c_id_idx_dict: dict[int, int]  # cell id to index mapping
    success: bool
    tour_idxs: list[int]
    tour_ids: list[int]

    def __init__(self, mcells: list[BaseCell]):
        self.ns = "BaseSeqOpt"
        self.lg = MyLogger(self.ns)
        self.mcells = mcells
        self.c_id_idx_dict = {c.id: i for i, c in enumerate(mcells)}
        self.success = False
        self.tour_idxs = []
        self.tour_ids = []

    def create_tour(self):
        raise NotImplementedError("This method should be overridden in subclasses.")


class SimpleSeqOpt(BaseSeqOpt):
    def __init__(self, mcells: list[BaseCell]):
        super().__init__(mcells)
        self.lg.info("using SimpleSeqOpt sequence optimization method.")

    def create_tour(self):
        self.tour_idxs = list(range(len(self.mcells)))
        self.tour_ids = [c.id for c in self.mcells]
        self.success = True


class DepthFirstSearch(BaseSeqOpt):
    def __init__(self, cells: list[BaseCell]):
        super().__init__(cells)
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
        neighbors = self.mcells[idx].adjacent_ids
        neighbors = [n for n in neighbors if n in self.c_ids]
        new_level = 0
        for neigh_id in neighbors:
            if not self.visited_ids_dict[neigh_id]:
                level = self.dfs(neigh_id, new_level)
                new_level = max(level, new_level)
        self.depth_ids_dict[c_id] = new_level
        return new_level + 1


class MILPSeq(BaseSeqOpt):
    def __init__(self, mcells: list[BaseCell], distances: np.ndarray):
        super().__init__(mcells)
        self.lg.info("using MILPSeq sequence optimization method.")

        # planning cells
        p_cells = [c for c in mcells if c.do_planning]
        p_idxs = [i for i, c in enumerate(mcells) if c.do_planning]
        self.p_cells = p_cells
        self.n_p_cells = len(p_idxs)

        #
        self.tour = None
        self.distm = np.zeros((self.n_p_cells, self.n_p_cells), dtype=float)

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
            print(f"[{self.ns}][ERROR]: No optimal solution found.")
            return

        c_ids = [c.id for c in self.p_cells]
        self.tour_ids = [c_ids[i] for i in self.tour]
        self.tour_idxs = [self.c_id_idx_dict[i] for i in self.tour_ids]

    def solve_tsp_milp(self) -> bool:
        solver = pywraplp.Solver.CreateSolver("SCIP")  # Use "CBC" for open-source

        n = self.n_p_cells
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
            self.tour = tour

        else:
            print("[solve_tsp_milp]: No optimal solution found.")
            return False
        #
        return True
