import numpy as np
from ortools.linear_solver import pywraplp


def solve_tsp(dist_t: np.ndarray, cities: np.ndarray, start_node: int = 0) -> tuple[list, float]:
    """TSP solver version 1.2 - from start_node to x
    """

    print("Using TSP solver version 1.2")
    n = len(cities)
    has_end_node = True

    # add end node if needed
    if has_end_node:
        end_node = 2
        print(f"Start node: {start_node}, End node: {end_node}")
        # add new city with big ditance to all other cities except end_node
        dist = np.array(dist_t, copy=True)
        # Add new row and column with inf values
        dist = np.pad(dist, ((0, 1), (0, 1)), constant_values=1000)
        dist[end_node, -1] = 0
        dist[n, end_node] = 0
        n += 1
    else:
        dist = dist_t
        print(f"Start node: {start_node}")

    #
    solver = pywraplp.Solver.CreateSolver('CBC')
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    # m1------------------------------------------------------------------------
    # 1. Each city entered at most once (except start)
    # 2. Each city exited at most once (except end)
    for i in range(n):
        solver.Add(solver.Sum([x[i, j] for j in range(n) if i != j]) == 1)
        solver.Add(solver.Sum([x[j, i] for j in range(n) if i != j]) == 1)

    # MTZ variables (u[i] = position of node i in the tour)
    u = [solver.NumVar(0, n-1, f'u[{i}]') for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and i != start_node and j != start_node:
                solver.Add(u[i] - u[j] + (n-1)*x[i, j] <= n-2)

    # Objective: Minimize total distance
    solver.Minimize(solver.Sum([dist[i][j] * x[i, j] for i in range(n) for j in range(n) if (i != j and j != start_node)]))

    # solve
    status = solver.Solve()

    # solution
    if status == pywraplp.Solver.OPTIMAL:
        # Reconstruct the path
        tour = [start_node]
        current = start_node
        while len(tour) < n:
            for j in range(n):
                if current != j and x[current, j].solution_value() > 0.5:
                    tour.append(j)
                    current = j
                    break

        # remove end node from the tour
        if has_end_node:
            tour.remove(n-1)

        # cost
        cost = solver.Objective().Value()
        return tour, cost
    else:
        return None, None
