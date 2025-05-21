
class Cell:
    def __init__(self, ind):
        self.ind = ind
        self.adjlist = []


class TestDFS:
    def __init__(self, mcells):
        self.cells = mcells
        self.visited = [False] * len(mcells)
        self.tour = []

    def run(self):
        # Start DFS from the first cell
        self.dfs(0)
        return self.tour

    def dfs(self, current: int):
        self.visited[current] = True
        self.tour.append(current)
        # Find all unvisited neighbors, sorted by distance (optional)
        neighbors = self.cells[current].adjlist
        for neighbor in neighbors:
            self.dfs(neighbor)


if __name__ == "__main__":
    # Create a test case with 5 cells
    cells = [Cell(i) for i in range(11)]
    # Manually set adjacency lists for testing
    cells[0].adjlist = [1, 2]
    cells[1].adjlist = [3, 4]
    cells[4].adjlist = [5]
    cells[2].adjlist = []
    cells[6].adjlist = [7]

    # Run the DFS test
    test_dfs = TestDFS(cells)
    tour = test_dfs.run()
    print("DFS Tour:", tour)
