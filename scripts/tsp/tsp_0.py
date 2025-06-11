import numpy as np
from ortools.linear_solver import pywraplp


def solve_tsp(dist: np.ndarray, cities: np.ndarray, start: int = 0) -> tuple[list, float]:
    """TSP solver version 0 - from 0 to 0 - loop

    Args:
        dist (np.ndarray): distance matrix
        cities (np.ndarray): xy of ciries, shape (n, 2)
        start (int, optional): starting city. Defaults to 0.

    Returns:
        tuple[list, float]: [tour, cost]
        tour: tour of cities, starting and ending at the start city [0]
        cost: total cost of the tour
    """

    print("Using TSP solver version 0")
    n = len(cities)

    solver = pywraplp.Solver.CreateSolver('CBC')
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    # 1. Each city entered at most once (except start)
    # 2. Each city exited at most once (except end)
    for i in range(n):
        solver.Add(solver.Sum([x[i, j] for j in range(n) if i != j]) == 1)
        solver.Add(solver.Sum([x[j, i] for j in range(n) if i != j]) == 1)

    # MTZ variables (u[i] = position of node i in the tour)
    u = [solver.NumVar(0, n-1, f'u[{i}]') for i in range(n)]
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                solver.Add(u[i] - u[j] + (n-1)*x[i, j] <= n-2)

    # Objective: Minimize total distance
    solver.Minimize(solver.Sum([dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j]))

    # solve
    status = solver.Solve()

    # solution
    if status == pywraplp.Solver.OPTIMAL:
        # Reconstruct the path
        tour: list = [0]
        current = 0
        while len(tour) < n:
            for j in range(n):
                if current != j and x[current, j].solution_value() > 0.5:
                    tour.append(j)
                    current = j
                    break
        tour.append(0)

        # cost
        cost: float = solver.Objective().Value()
        return tour, cost
    else:
        return None, None
