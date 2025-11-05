""" Segmentation Handler Module."""
from scripts.base.logger import MyLogger
from scripts.base.configs import PipeLineConfig
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.segmentation.cell_base import CellBase
from scripts.segmentation.cell_processed import PCell
from scripts.segmentation.segmentation_base import SegmentationBase


class SegmentationHandler:
    """ High-level Segmentation Interface.
        Handles segmentation of a zone. Results in processed cells.
    """

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("SegmentationHandler")

        # configs
        self.dcfgs = PipeLineConfig()

        # zone data
        self.zmd: ZoneMapData = zmd

        # create map polygons object for the zone
        self.map_polygons: MapPolygons = MapPolygons(self.zmd)

        # main segmentor (to be created in the run method based on configs)
        self.segmentor: SegmentationBase = None

        # attributes
        self.bcells: list[CellBase] = []        # base cells from segmentation
        self.cells: list[PCell] = []            # processed cells
        self.s_cells: list[PCell] = []          # sorted processed cells
        self.n_cells: int = 0                   # number of cells

    def run(self) -> None:
        """ Run segmentation.

        Steps:
            1. select segmentation method from configs.
            2. create segmentor object and run segmentation.
            3. create processed cells from base cells.
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

        # resulting base cells from segmentor
        self.bcells = self.segmentor.cells
        self.n_cells = self.segmentor.n_cells

        # create processed cells
        self.cells = [PCell(cell, self.zmd) for cell in self.bcells]
