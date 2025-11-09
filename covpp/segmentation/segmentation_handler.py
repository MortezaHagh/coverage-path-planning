"""Segmentation handler module for managing zone decomposition into cells.

This module implements the Handler pattern to dynamically load and execute
different segmentation algorithms based on configuration.
"""
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.planning.map_polygons import MapPolygons
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.cell_processed import PCell
from covpp.segmentation.merge_cells import MergeCells
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.simple_segmentation.segmentation_simple import Segmentation as SegmentationSimple


class SegmentationHandler:
    """High-level interface for zone segmentation into coverage cells.

    This handler dynamically loads the appropriate segmentation algorithm
    based on the pipeline configuration and converts raw cells into
    processed cells ready for coverage planning.

    Args:
        zmd: Zone map data object.

    Attributes:
        zmd: Zone map data object.
        map_polygons: Polygon representation of the workspace.
        segmentor: The actual segmentation algorithm instance.
        bcells: Base cells from segmentation algorithm.
        cells: Processed cells ready for planning.
        s_cells: Sorted processed cells.
        n_cells: Number of cells created.
        dcfg: Pipeline configuration object.

    Supported Methods:
        - "simple": Single cell covering entire zone
        - "voronoi": Voronoi-based decomposition
        - Custom methods can be added by extending SegmentationBase
    """

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("SegmentationHandler")

        # configs
        self.dcfg = PipeLineConfig()

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
        """Execute segmentation pipeline using configured method.

        This method:
            1. Loads the segmentation algorithm based on configuration
            2. Executes the segmentation to create base cells
            3. Converts base cells to processed cells (PCell objects)

        Raises:
            ValueError: If segmentation method is not recognized.
        """

        # segmentation method
        segmentation_method = self.dcfg.segmentation_method

        # create segmentor
        if segmentation_method == "simple":
            self.segmentor = SegmentationSimple(self.zmd, self.map_polygons)
        elif segmentation_method == "voronoi":
            from covpp.segmentation.voronoi_segmentation.segmentation_voronoi import Segmentation
            self.segmentor = Segmentation(self.zmd, self.map_polygons)
        else:
            raise ValueError(f"Segmentation method '{segmentation_method}' not recognized.")

        # initiate segmentation
        self.segmentor.run()

        # resulting base cells from segmentor
        bcells = self.segmentor.cells
        n_bcells = self.segmentor.n_cells

        if n_bcells == 0:
            self.lg.warn("No cells were created by the segmentation algorithm.")
            self.lg.warn("Trying simple segmentation.")
            self.segmentor = SegmentationSimple(self.zmd, self.map_polygons)
            self.segmentor.run()
            bcells = self.segmentor.cells
            n_bcells = self.segmentor.n_cells
            if n_bcells == 0:
                raise ValueError("No cells were created even by simple segmentation.")

        # merge cells
        merger = MergeCells(bcells)
        self.bcells = merger.updated_cells
        self.n_cells = len(self.bcells)

        # check if ids are unique
        cell_ids = [cell.id for cell in self.bcells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Cell IDs are not unique after merging.")

        # update cell ids to be consecutive
        cell_id_indx_map = {cell.id: i for i, cell in enumerate(self.bcells)}
        for cell in self.bcells:
            cell.id = cell_id_indx_map[cell.id]
            # update adjacent ids
            cell.adjacent_ids = [cell_id_indx_map[adj_id] for adj_id in cell.adjacent_ids]

        # create processed cells
        self.cells = [PCell(cell, self.zmd) for cell in self.bcells]
