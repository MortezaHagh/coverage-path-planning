from ortools.linear_solver import pywraplp


def solve_tsp(dist, cities, start_node=0):
    print("Using TSP solver version 1.1")
    n = len(cities)

    solver = pywraplp.Solver.CreateSolver('CBC')
    x = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x[i, j] = solver.BoolVar(f'x[{i},{j}]')

    # Out-degree and in-degree constraints
    for i in range(n):
        if i == start_node:
            # Start node: out-degree 1, in-degree 0
            solver.Add(solver.Sum([x[start_node, j] for j in range(n) if start_node != j]) == 1)
            solver.Add(solver.Sum([x[j, start_node] for j in range(n) if start_node != j]) == 0)
        else:
            # Other nodes: in-degree <= 1, out-degree <= 1
            solver.Add(solver.Sum([x[i, j] for j in range(n) if i != j]) <= 1)
            solver.Add(solver.Sum([x[j, i] for j in range(n) if i != j]) <= 1)
    # Ensure exactly one end node (node with out-degree 0 and in-degree 1, not start_node)
    solver.Add(solver.Sum([1 - solver.Sum([x[i, j] for j in range(n) if i != j]) for i in range(n) if i != start_node]) == 1)

    # MTZ variables (u[i] = position of node i in the tour)
    u = [solver.NumVar(0, n-1, f'u[{i}]') for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and i != start_node and j != start_node:
                solver.Add(u[i] - u[j] + (n-1)*x[i, j] <= n-2)

    # Objective: Minimize total distance
    solver.Minimize(solver.Sum([dist[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j]))

    # Solve
    status = solver.Solve()

    # solution
    if status == pywraplp.Solver.OPTIMAL:
        # Reconstruct the path
        tour = [start_node]
        current = start_node
        while True:
            found = False
            for j in range(n):
                if current != j and x[current, j].solution_value() > 0.5:
                    tour.append(j)
                    current = j
                    found = True
                    break
            if not found:
                break

        # cost
        cost = solver.Objective().Value()
        return tour, cost
    else:
        return None, None
