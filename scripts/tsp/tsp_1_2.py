from ortools.linear_solver import pywraplp


def solve_tsp(dist, cities, start_node=0):
    n = len(cities)

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

        # cost
        cost = solver.Objective().Value()
        return tour, cost
    else:
        return None, None
