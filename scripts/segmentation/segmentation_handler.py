from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfigs
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.segmentation.cell_base import CellBase
from scripts.segmentation.cell_processed import PCell
from scripts.segmentation.segmentation_base import SegmentationBase


class SegmentationHandler:
    """ High-level Segmentation Interface.
    """

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("SegmentationHandler")

        # configs
        self.dcfgs = DistrictConfigs()

        # zone data
        self.zmd: ZoneMapData = zmd
        self.map_polygons: MapPolygons = MapPolygons(self.zmd)

        #
        self.bcells: list[CellBase] = []     # cells from segmentation
        self.cells: list[PCell] = []     # processed cells
        self.s_cells: list[PCell] = []   # sorted cells
        self.seq_idxs: list[int] = None     # resulted sequence of cell indices
        self.n_cells: int = 0               # number of cells

        # main segmentor (to be created in the run method based on configs)
        self.segmentor: SegmentationBase = None

    def run(self) -> None:
        """ Run segmentation.
        """

        # segmentation method
        segmentation_method = self.dcfgs.segmentation_method

        # create segmentor
        if segmentation_method == "simple":
            from scripts.segmentation.simple_segmentation.segmentation_simple import Segmentation
            self.segmentor = Segmentation(self.zmd, self.map_polygons)
        elif segmentation_method == "voronoi":
            from scripts.segmentation.voronoi_decomposition.segmentation_voronoi import Segmentation
            self.segmentor = Segmentation(self.zmd, self.map_polygons)
        else:
            raise ValueError(f"Segmentation method '{segmentation_method}' not recognized.")

        # initiate segmentation
        self.segmentor.run()

        # resulting base cells
        self.bcells = self.segmentor.cells
        self.n_cells = self.segmentor.n_cells

        # create processed cells
        self.cells = [PCell(cell, self.zmd) for cell in self.bcells]
