import numpy as np
from scripts.planning.boustrophedon.cell import Cell
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.optimize_mcells_seq_1 import OptimCellsSeq
from scripts.planning.boustrophedon.decomposition_bs_mh import MHDecomposition as CellDecomposition


class DecompositionInterface:
    ns: str
    zmd: ZoneMapData
    decomposed: CellDecomposition
    mcells: list[Cell]
    smcells: list[Cell]
    seq_idxs: list[int]
    n_mcells: int

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        self.ns = "DecompositionInterface"
        self.zmd = zmd

        #
        self.mcells = []
        self.smcells = []
        self.seq_idxs = None
        self.n_mcells = 0

        # cell decomposition
        self.decomposed = CellDecomposition(zmd.resolution, zmd.result_path)
        # set bounding polygon
        self.decomposed.set_bounding_polygon(map_polygons.b_poly)
        # set obstacle polygons
        self.decomposed.set_obstacle_polygons(map_polygons.obs_polys)

        # initiate decomposition
        self.decomposed.initiate()

        # resulting cells
        self.mcells = self.decomposed.mcells
        self.n_mcells = self.decomposed.n_mcells

    def create_sequence(self, distances: np.ndarray = None):
        if len(self.mcells) == 0:
            return
        elif len(self.mcells) == 1:
            self.smcells = self.mcells
            return

        # create optimized sequence of cells
        opt_seq = OptimCellsSeq(mcells=self.mcells, distances=distances)
        self.seq_idxs = opt_seq.tour_idxs
        self.smcells = [self.mcells[i] for i in opt_seq.tour_idxs]
