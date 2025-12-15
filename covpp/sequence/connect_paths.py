"""Path connection module. """
import numpy as np
from covpp.base.logger import MyLogger
from covpp.base.geometry import MyPath, MyPoint
from covpp.segmentation.cell_planning import PCell
from covpp.path_planning.path_in_map import FindPathInMap


class ConnectPaths:
    """Generate transition paths between consecutive coverage cells.

    Uses A* pathfinding algorithm to find collision-free paths connecting
    the end point of one cell's coverage path to the start point of the
    next cell in the sequence.
    """

    def __init__(self, map_ws: np.ndarray, g_xy: tuple[int, int], ns: str = ""):
        self.lg = MyLogger(f"ConnectPaths-{ns}")

        # find path in map object

        self.fp = FindPathInMap(map_ws, g_xy).pp

        # all connections between cells
        self.all_connection_paths: np.ndarray = np.array([])    # 2D array of paths between cells
        self.all_distances: np.ndarray | None = None            # 2D array of distances between cells
        self.c_paths_px: list[MyPath] = []                      # list of connection paths for a given sequence
        self.n_cells: int = None                                # number of cells

    def initialize(self, cells: list[PCell]):
        """ Initialize connection paths and distances matrices based on the number of cells.
            - i, j are cell indexes in the cells list
        """
        # cell_ids = [cell.id for cell in cells]
        # n = max(cell_ids)
        n = len(cells)
        self.n_cells = n
        self.all_connection_paths = np.full((n, n), None, dtype=object)
        self.all_distances = np.full((n, n), np.inf, dtype=float)

    def find_all_connections(self, cells: list[PCell]):
        """ Find all connection paths between cells (based on their coverage paths endpoints) and store them in a matrix. """
        self.lg.info2("find all connections paths")
        n = self.n_cells
        all_connections = np.full((n, n), None, dtype=object)

        for i in range(n):
            for j in range(i + 1, n):
                if cells[i].end_point is None or cells[j].start_point is None:
                    all_connections[i, j] = None
                    all_connections[j, i] = None
                else:
                    all_connections[i, j] = self.find_path(cells[i].end_point, cells[j].start_point)
                    all_connections[j, i] = self.find_path(cells[j].end_point, cells[i].start_point)
        self.all_connection_paths = all_connections

    def find_all_distances(self):
        """ Calculate distances based on connection paths and store them in a matrix. """
        self.lg.info2("find all distances")
        n = self.n_cells
        all_distances = np.full((n, n), np.inf, dtype=float)
        for i in range(n):
            for j in range(n):
                path: MyPath = self.all_connection_paths[i, j]
                if path is None:
                    continue
                all_distances[i, j] = np.sum(np.linalg.norm(np.diff(path.xy, axis=0), axis=1))
        self.all_distances = all_distances

    def connect_paths(self, paths_px: list[MyPath], seq_idxs: list[int] = None) -> None:
        """ Connect paths based on the given sequence of cell indices.
            - If seq_idxs is None, connect paths in the given order.
            - If path between two cells is not found, create it.
        """
        c_paths = []  # list of connection paths

        # check if seq_idxs (sequence) is given
        if not seq_idxs:
            for ip in range(len(paths_px) - 1):
                p0 = paths_px[ip].points[-1]
                p1 = paths_px[ip+1].points[0]
                c_paths.append(self.find_path(start=p0, end=p1))
        else:
            for ip, idx in enumerate(seq_idxs[:-1]):
                path = self.all_connection_paths[idx, seq_idxs[ip+1]]
                if path is None:
                    p0 = paths_px[ip].points[-1]    # end of current path
                    p1 = paths_px[ip+1].points[0]   # start of next path
                    path = self.find_path(start=p0, end=p1)
                c_paths.append(path)
        self.c_paths_px = c_paths

    def find_path(self, start: MyPoint, end: MyPoint) -> MyPath:
        """ Find path between two points using the FindPathInMap class. """
        return self.fp.find_path(start, end)

    def update_connection_2_cells(self, start_idx: int, end_idx: int, start: MyPoint, end: MyPoint) -> float:
        """ Update (recalculate) connection path and distance between two cells."""
        if start is None or end is None:
            self.lg.warn("Start or end index is None.")
            return float('inf')
        path = self.fp.find_path(start, end)
        self.all_connection_paths[start_idx, end_idx] = path
        length = np.sum(np.linalg.norm(np.diff(path.xy, axis=0), axis=1))
        self.all_distances[start_idx, end_idx] = length
        return length
