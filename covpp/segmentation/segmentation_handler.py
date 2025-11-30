"""Segmentation handler module for managing zone decomposition into cells.

This module implements the Handler pattern to dynamically load and execute
different segmentation algorithms based on configuration.
"""
from importlib.metadata import entry_points
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.cell_processed import PCell
from covpp.segmentation.merge_cells import MergeCells
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.simple_segmentation.segmentation_simple import Segmentation as SegmentationSimple


class SegmentationHandler:
    """High-level interface for zone segmentation into coverage cells.

    This handler dynamically loads the appropriate segmentation algorithm
    based on the pipeline configuration and converts raw cells into
    processed cells ready for coverage planning.

    Args:
        zmd (ZoneMapData): Zone map data object.

    Attributes:
        zmd (ZoneMapData): Zone map data object.
        map_polygons (MapPolygons): Polygon representation of the workspace.
        segmentor (SegmentationBase): The actual segmentation algorithm instance.
        bcells (list[CellBase]): Base cells from segmentation algorithm.
        cells (list[PCell]): Processed cells ready for planning.
        s_cells (list[PCell]): Sorted processed cells.
        n_cells (int): Number of cells created.
        dcfg (PipeLineConfig): Pipeline configuration object.

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
        self.map_polygons = MapPolygons(map_ws=self.zmd.map_ws, xy=self.zmd.global_xy)

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

        # ----------------------------------------------------------------------
        # create segmentor based on segmentation method
        segmentation_method = self.dcfg.segmentation_method
        segmentors = entry_points(group="segmentors")
        if segmentation_method not in segmentors.names:
            raise ValueError(f"Segmentation method '{segmentation_method}' not recognized.")
        self.segmentor: SegmentationBase = segmentors[segmentation_method].load()(self.zmd, self.map_polygons)

        # run segmentation =====================================================
        self.segmentor.run()
        # ======================================================================

        # process and merge resulting cells into PCell objects
        self.process_cells()

        # validate created cells
        self.validate_cells()

    def process_cells(self) -> None:
        # resulting base cells from segmentor
        bcells = self.segmentor.cells
        n_bcells = self.segmentor.n_cells

        # if no cells were created, try simple segmentation
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

        # check if ids are unique
        cell_ids = [cell.id for cell in self.bcells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Cell IDs are not unique after merging.")

        # create processed cells
        self.cells = [PCell(cell, self.zmd) for cell in self.bcells]
        self.n_cells = len(self.bcells)

    def validate_cells(self) -> None:
        """ Validate the created cells for planning readiness. """

        if len(self.cells) == 0 or self.cells is None:
            raise ValueError("No processed cells available for validation.")

        cell_validity = [cell.validate_cell() for cell in self.cells]
        if not all(cell_validity):
            raise ValueError("One or more cells failed validation checks.")
