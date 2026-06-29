"""Segmentation handler module for managing zone decomposition into cells.

This module implements the Handler pattern to dynamically load and execute
different segmentation algorithms based on configuration.
"""
import os
import cv2 as cv
import numpy as np
from importlib.metadata import entry_points
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.base.base_utils import get_n_colors_cv
from covpp.district.zone_map_data import ZoneMapData
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.cell_planning import PCell
from covpp.segmentation.merge_cells import MergeCells
from covpp.segmentation.map_polygons import MapPolygons
from covpp.segmentation.segmentation_base import SegmentationBase
from covpp.segmentation.simple_segmentation.segmentation_simple import Segmentation as SegmentationSimple
from covpp.segmentation.segmentation_improvement.ebpts_graph_segmentation import EspecialBranchPointsSegmentation


class SegmentationHandler:
    """High-level interface for zone segmentation into coverage cells.

    This handler dynamically loads the appropriate segmentation algorithm
    based on the pipeline configuration and converts raw cells into
    processed cells ready for coverage planning.

    Result:
        cells (list[PCell])

    Args:
        zmd (ZoneMapData): Zone map data object.

    Attributes:
        zmd (ZoneMapData): Zone map data object.
        map_polygons (MapPolygons): Polygon representation of the workspace.
        segmentor (SegmentationBase): The actual segmentation algorithm instance.
        bcells (list[CellBase]): Base cells from segmentation algorithm.
        cells (list[PCell]): Processed cells ready for planning.
        n_cells (int): Number of cells created.

    Supported Methods:
        - "simple": Single cell covering entire zone
        - "voronoi": Voronoi-based decomposition
        - Custom methods can be added by extending SegmentationBase
    """

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("SegmentationHandler")

        # zone data
        self.zmd: ZoneMapData = zmd

        # create map polygons object for the zone
        self.map_polygons = MapPolygons(map_ws=self.zmd.map_ws, xy=self.zmd.global_xy)

        # main segmentor (to be created in the run method based on configs)
        self.segmentor: SegmentationBase = None

        # attributes
        self.cells_l1: list[CellBase] = []      # level 1 cells from segmentation
        self.bcells: list[CellBase] = []        # base cells from segmentation
        self.cells: list[PCell] = []            # planning cells
        self.n_cells: int = 0                   # number of cells

    def run(self) -> None:
        """Execute segmentation pipeline using configured method.

        This method:
            1. Loads the segmentation algorithm based on configuration
            2. Executes the segmentation to create base cells
            3. Converts base cells to planning cells (PCell objects) by processing and merging
            4. Validates the created cells for planning readiness

        Raises:
            ValueError: If segmentation method is not recognized.
        """

        # ----------------------------------------------------------------------
        # create segmentor based on segmentation method
        segmentation_method = PipeLineConfig.segmentation_method
        segmentors = entry_points(group="segmentors")
        if segmentation_method not in segmentors.names:
            raise ValueError(f"Segmentation method '{segmentation_method}' not recognized.")
        self.segmentor: SegmentationBase = segmentors[segmentation_method].load()(self.zmd, self.map_polygons)
        # ----------------------------------------------------------------------

        # run segmentation =====================================================
        self.segmentor.run()
        self.cells_l1 = self.segmentor.cells
        # ======================================================================

        #
        self.improve_segmentation()

        # process and merge resulting cells into PCell objects
        self.process_cells()

        # validate created cells
        self.validate_cells()

        # plot segmentation results
        self.add_cells_to_result_maps()
        self.save_segmentation_map()

    def improve_segmentation(self) -> None:
        improver = EspecialBranchPointsSegmentation(self.zmd, self.cells_l1)
        improver.run()

    def process_cells(self) -> None:
        """ Convert base cells from segmentor into planning PCell objects. """
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
        # self.bcells = bcells

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

    # def plot_segmentation(self) -> None:
    #     """Plot segmentation results."""
    #     from matplotlib import pyplot as plt
    #     plt.close('all')
    #     fig, ax = plt.subplots()
    #     ax.set_aspect('equal')
    #     for cell in self.cells:
    #         poly = cell.poly
    #         x, y = poly.exterior.xy
    #         ax.plot(x, y)
    #         # text at centroid
    #         centroid = poly.centroid
    #         ax.text(centroid.x, centroid.y, f"{cell.id}", fontsize=12, ha='center', va='center')

    #     ax.set_title("Segmented Polygons")
    #     plt.show()

    def add_cells_to_result_maps(self) -> None:
        """ Add cell boundaries and fills to the zone's planning and segmentation maps for visualization. """
        colors = get_n_colors_cv(self.n_cells)
        for i, cell in enumerate(self.cells):
            pp = cell.points
            pp = np.array(pp, dtype=np.int32)  # Ensure pp is a numpy array of integers
            # add cells to map_planning
            map_p = cv.polylines(self.zmd.map_planning, [pp], isClosed=True, color=(100, 100, 100), thickness=2)
            map_p_fill = self.zmd.map_planning.copy()
            cv.fillPoly(map_p_fill, [pp], color=colors[i])
            alpha = 0.1
            self.zmd.map_planning = cv.addWeighted(map_p_fill, alpha, map_p, 1 - alpha, 0)
            # add cells to map_segmentation
            map_s = cv.polylines(self.zmd.map_segmentation, [pp], isClosed=True, color=(100, 100, 100), thickness=5)
            map_s_fill = self.zmd.map_segmentation.copy()
            cv.fillPoly(map_s_fill, [pp], color=colors[i])
            alpha = 0.1
            self.zmd.map_segmentation = cv.addWeighted(map_s_fill, alpha, map_s, 1 - alpha, 0)

            # # add rectangle
            # if cell.rect_like:
            #     rp = cell.rectangle_poly.exterior.coords.xy
            #     rpoints = np.array(list(zip(rp[0], rp[1])), dtype=np.int32)
            #     self.zmd.map_planning = cv.polylines(self.zmd.map_planning, [rpoints], isClosed=True, color=(0, 0, 0), thickness=2)
            #     self.zmd.map_segmentation = cv.polylines(self.zmd.map_segmentation, [rpoints], isClosed=True, color=(0, 0, 0), thickness=2)

    def save_segmentation_map(self) -> None:
        """ Save the segmentation map to file. """
        if not PipeLineConfig.do_save_maps:
            return
        # PNG parameters for lossless compression
        png_params = [cv.IMWRITE_PNG_COMPRESSION, 0]
        m_name_s = "map_segmentation.png"
        map_s_path = os.path.join(self.zmd.segmentation_method_dir, m_name_s)
        cv.imwrite(map_s_path, self.zmd.map_segmentation, png_params)
