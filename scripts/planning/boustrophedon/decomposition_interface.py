from scripts.planning.map_polygons import MapPolygons
from scripts.planning.boustrophedon.decomposition_bs_mh import MHDecomposition as CellDecomposition


class DecompositionInterface:
    ns: str
    decomposed: CellDecomposition
    n_mcells: int

    def __init__(self, map_polygons: MapPolygons, reslolution: float):
        self.ns = "DecompositionInterface"
        self.decomposed = CellDecomposition(reslolution)
        self.n_mcells = 0

        #
        self.decomposed.set_bounding_polygon(map_polygons.b_poly)
        # set obstacle polygons
        self.decomposed.set_obstacle_polygons(map_polygons.obs_polys)

        # initiate decomposition
        self.decomposed.initiate()
        self.n_mcells = self.decomposed.n_mcells
