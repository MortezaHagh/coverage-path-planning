import matplotlib.pyplot as plt
import numpy as np
m = 2
if m == 0:
    from tsp_0 import solve_tsp
elif m == 1:
    from tsp_1_1 import solve_tsp
elif m == 2:
    from tsp_1_2 import solve_tsp


class TSPSolver:
    def __init__(self, cities_t, start=0, plot=True):
        self.cities = cities_t
        self.n = len(cities_t)
        self.start = start
        self.do_plot = plot
        self.distances = []
        self.tour = []
        self.success = True

        #
        self.cal_distance_matrix()
        self.solve()
        self.eval()
        self.plot_sol()

    def solve(self):
        s_tour, s_cost = solve_tsp(self.distance_matrix, self.cities, self.start)
        self.tour = s_tour
        self.cost = s_cost
        if self.tour is None:
            self.success = False

    def eval(self):
        if not self.success:
            print("No solution found.")
            return
        print(f"Cost: {self.cost}")
        print(f"Tour: {self.tour}")

    def cal_distance_matrix(self):
        dist = [[self.distance(i, j) for j in range(self.n)] for i in range(self.n)]
        self.distance_matrix = dist

    def distance(self, i, j):
        return np.linalg.norm(self.cities[i] - self.cities[j])

    def plot_sol(self):
        if not (self.do_plot and self.success):
            return
        citiesp = self.cities
        tourp = self.tour
        #
        plt.figure(figsize=(6, 6))
        plt.plot(citiesp[:, 0], citiesp[:, 1], 'o', color='red')
        for i, (x_, y_) in enumerate(citiesp):
            plt.text(x_, y_, str(i), fontsize=12, ha='right')
        for i in range(len(tourp)-1):
            plt.plot([citiesp[tourp[i], 0], citiesp[tourp[i+1], 0]],
                     [citiesp[tourp[i], 1], citiesp[tourp[i+1], 1]], 'b-')
        plt.title('TSP Solution')
        plt.show()


# Example usage:
if __name__ == "__main__":
    cities = np.array([
        [0, 0],
        [1, 5],
        [5, 2],
        [6, 6],
        [8, 3]
    ])
    TSPSolver(cities, start=3, plot=True)
