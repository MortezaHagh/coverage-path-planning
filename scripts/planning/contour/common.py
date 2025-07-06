import numpy as np


class VPoint:
    """ Voronoi point class representing a vertex in the Voronoi diagram. """

    def __init__(self, v: np.ndarray, idx: int):
        self.idx: int = idx                         # index in the Voronoi vertice
        self.v: np.ndarray = v                      # coordinates of the Voronoi vertex
        self.free: bool = False
        self.feasible: bool = False
        self.critical: bool = False                 # critical Voronoi vertex
        self.has_inf_adj: bool = False              # has adjacent Voronoi vertices with infinite distance
        self.in_bd_poly: bool = False               # inside the outer bounding polygon 1
        #
        self.basis_idxs: np.ndarray = None          # two closest basis points
        self.basis_points: np.ndarray = None        # two closest basis points in the Voronoi diagram
        self.basis_dists: np.ndarray = None         # distances to the basis points
        self.clearance: float = np.inf              # clearance to the obstacles
        self.adjacent_vp_idxs: list[int] = []       # indices of adjacent Voronoi points
        self.adjacent_vp_dists: list[float] = []    # distances to adjacent Voronoi points
        # #
        self.n_feas_ridges: int = 0                 # number of feasible adjacent Voronoi points
        self.ridge_vps: list[VPoint] = []           # Free Voronoi vertex that is a ridge point for this Voronoi vertex

    def set_basis_points(self, basis_idxs: np.ndarray, points: np.ndarray):
        """ Set the basis points (closest obstacle points) for the Voronoi point and calculate their distances."""
        if basis_idxs is None or len(basis_idxs) < 2:
            raise ValueError("Basis indices must be provided and contain at least two indices.")
        self.basis_idxs = basis_idxs
        self.basis_points = points[basis_idxs]
        basis_dist = np.linalg.norm(self.basis_points - self.v, axis=1)
        self.set_basis_dists(basis_dist)

    def set_basis_dists(self, dists: np.ndarray):
        if dists is None or len(dists) < 1:
            raise ValueError("Basis distances must be provided and contain at least one distances.")
        self.basis_dists = dists
        self.clearance = round(np.sum(dists), 2)

    def set_adjacent_vp_idxs(self, adj_vp_idxs: np.ndarray):
        """ Set the indices of adjacent Voronoi points. 
            adjacents based on the distance and min-max number of adjacents"""
        self.adjacent_vp_idxs = adj_vp_idxs

    def set_adjacents_vp_dists(self, dists: np.ndarray):
        self.adjacent_vp_dists = dists


class CriticalPoint:
    """ Critical point class representing a Voronoi point with additional properties. """

    def __init__(self, vp: VPoint, sid: int = None):
        if vp is None:
            raise ValueError("Voronoi point must be provided.")
        self.id = sid
        self.vp: VPoint = vp            # Voronoi point associated with this critical point
        self.bs1_idx: int = None        # index of the first basis point
        self.bs2_idx: int = None        # index of the second basis point
        self.bs1_vp: VPoint = None      # basis point 1
        self.bs2_vp: VPoint = None      # basis point 2
        self.passed_12: bool = False    # whether this critical point has been passed from basis point 1 to basis point 2
        self.passed_21: bool = False    # whether this critical point has been passed from basis point 2 to basis point 1
        self.n_passed: int = 0          # number of passings through this critical point in the ring creation
        self.skip: bool = False         # skip this critical point in the ring creation

        self.set_basis_idxs()

    def set_basis_idxs(self):
        idx_1 = self.vp.basis_idxs[0]
        idx_2 = self.vp.basis_idxs[1]
        if idx_1 < idx_2:
            self.bs1_idx = idx_1
            self.bs2_idx = idx_2
            self.bs1_vp = self.vp.basis_points[0]
            self.bs2_vp = self.vp.basis_points[1]
        else:
            self.bs1_idx = idx_2
            self.bs2_idx = idx_1
            self.bs1_vp = self.vp.basis_points[1]
            self.bs2_vp = self.vp.basis_points[0]
