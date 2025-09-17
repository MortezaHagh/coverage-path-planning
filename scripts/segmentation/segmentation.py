from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfigs
from scripts.segmentation.cell_base import CellBase
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData


class SegmentationBase:
    """ Base class for Segmentation.  
    """

    def __init__(self, zmd: ZoneMapData, map_polygons: MapPolygons):
        self.ns: str = "SegmentationBase"

        # zone map data
        self.zmd: ZoneMapData = zmd

        # configs
        self.dcfgs: DistrictConfigs = DistrictConfigs()

        # map polygons
        self.map_polygons: MapPolygons = map_polygons

        # cells
        self.cells: list[CellBase] = []
        self.n_cells: int = 0

    def run(self):
        """ Run segmentation. 
            to be implemented in child classes. 
        """
        raise NotImplementedError


class SegmentationInterface:
    """ Segmentation Interface.  
    """

    def __init__(self, zmd: ZoneMapData):
        self.ns: str = "SegmentationInterface"
        self.lg = MyLogger(self.ns)

        # zone data
        self.zmd: ZoneMapData = zmd
        self.map_polygons: MapPolygons = MapPolygons(self.zmd)

        #
        self.cells: list[CellBase] = []     # cells from segmentation
        self.s_cells: list[CellBase] = []   # sorted cells
        self.seq_idxs: list[int] = None     # resulted sequence of cell indices
        self.n_cells: int = 0               # number of cells

        # main segmentor (to be created in the run method based on configs)
        self.segmentor: SegmentationBase = None

    def run(self):
        """ Run segmentation.
        """

        # segmentation method
        dcfgs = DistrictConfigs()
        segmentation_method = dcfgs.segmentation_method

        # create segmentor
        if segmentation_method == "simple":
            from scripts.segmentation.segmentation_simple import Segmentation
            self.segmentor = Segmentation(self.zmd, self.map_polygons)
        elif segmentation_method == "voronoi":
            from scripts.segmentation.voronoi_decomposition.segmentation_voronoi import Segmentation
            self.segmentor = Segmentation(self.zmd, self.map_polygons)
        else:
            raise ValueError(f"Segmentation method '{segmentation_method}' not recognized.")

        # initiate segmentation
        self.segmentor.run()

        # resulting cells
        self.cells = self.segmentor.cells
        self.n_cells = self.segmentor.n_cells
