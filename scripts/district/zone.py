"""Zone data management module for zone-level coverage planning operations."""
import os
import time
import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.sequence.sequence import Sequence
from scripts.segmentation.cell_processed import PCell
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import PipeLineConfig, PlanningConfig
from scripts.segmentation.segmentation_handler import SegmentationHandler
from scripts.planning.coverage_path_plan_handler import CoveragePathPlanHandler
from scripts.evaluation.zone_evaluation_handler import ZoneEvaluationHandler
from scripts.evaluation.zone_evaluation_base import ZoneStatsBase


class ZoneData:
    """Manage zone-level data and execute complete coverage planning pipeline.

    This class coordinates all zone-level operations including segmentation,
    coverage path planning, path sequencing, and performance evaluation.

    Args:
        ind: Zone index identifier.
        zmd: Zone map data object containing workspace and map information.

    Attributes:
        ind: Zone index identifier.
        zmd: Zone map data object.
        segmentation_handler: Handler for space segmentation operations.
        cells: List of processed cells from segmentation.
        cov_path_planner: Handler for coverage path planning operations.
        sequence: Handler for path sequencing and connection operations.
        evaluator: Handler for zone performance evaluation.
        complete_path: Complete connected path (future implementation).
        stats: Zone-level statistics object.
        dcfg: Pipeline configuration object.
        pcfg: Planning configuration object.

    Example:
        >>> zone = ZoneData(ind=0, zmd=zone_map_data)
        >>> zone.zone_planning_pipeline()
    """

    def __init__(self, ind: int, zmd: ZoneMapData):
        self.lg = MyLogger(f"ZoneData-{ind}")

        # attributes
        self.ind: int = ind
        self.zmd: ZoneMapData = zmd

        # # # configs
        self.dcfg: PipeLineConfig = PipeLineConfig()
        self.pcfg: PlanningConfig = PlanningConfig()

        # # # segmentation
        self.segmentation_handler: SegmentationHandler = SegmentationHandler(self.zmd)
        self.cells: list[PCell] = []

        # # # coverage path planner
        self.cov_path_planner: CoveragePathPlanHandler = CoveragePathPlanHandler(self.zmd)

        # # # sequence
        self.sequence: Sequence = Sequence(self.zmd)

        # # # evaluator
        self.evaluator: ZoneEvaluationHandler = ZoneEvaluationHandler(self.zmd)

        # resulting data
        self.complete_path: np.ndarray = None  # todo

        # statistics
        self.stats: ZoneStatsBase = None

        # initialize
        self.initialize()

    def initialize(self):
        """Initialize zone data by saving map data and creating initial planning map."""
        self.zmd.save_data()
        self.create_map_planning()

    def zone_planning_pipeline(self):
        """Execute complete coverage planning pipeline for the zone.

        Pipeline Steps:
            1. Segmentation: Divide zone into cells
            2. Coverage Planning: Generate paths for each cell
            3. Sequencing: Optimize cell order and connect paths
            4. Evaluation: Compute performance statistics
            5. Post-processing: Update visualization maps

        This method orchestrates all planning operations and measures execution time.
        """
        start_time = time.time()

        # ************ segmentation
        self.lg.info1(f"Step: running segmentation using method '{self.dcfg.segmentation_method}'")
        self.segmentation_handler.run()
        self.cells = self.segmentation_handler.cells

        # ************ coverage planning
        self.lg.info1(f"Step: running coverage path planning using method '{self.dcfg.covpp_method}'")
        self.cov_path_planner.plan(cells=self.cells)

        # ************ sequence
        self.lg.info1(f"Step: creating sequence and full path using method '{self.dcfg.sequence_method}'")
        self.sequence.run(cells=self.cells, map_polygons=self.segmentation_handler.map_polygons)

        end_time = time.time()
        process_time = end_time - start_time

        # ************ evaluate statistics
        self.evaluator.evaluate(paths=self.sequence.paths, duration=process_time)
        self.stats = self.evaluator.stats

        # ************
        self.lg.info1("Step: zone post planning")
        self.post_planning()

    def post_planning(self):
        """ post planning operations:
            1. update map_planning with coverage path and sequence.
            2. save map_planning.
            3. save map showing cleaned area and uncleaned area.
        """
        self.save_map_planning()
        self.save_map_cleaning_data()

    # ==================================================================================================================

    def create_map_planning(self):
        """ create initial map_planning with zone boundary. """
        map_planning = self.zmd.map_planning
        vertices = self.zmd.vertices
        map_planning = cv.polylines(map_planning, [vertices], isClosed=True, color=(100, 0, 150), thickness=2)
        self.zmd.map_planning = map_planning
        self.zmd.map_segmentation = map_planning.copy()

    def save_map_planning(self):
        """ save zone map_planning and map_segmentation to files. """
        if not self.dcfg.save_map_planning:
            return
        # save map planning
        m_name = "map_planning.png"
        map_hl_path = os.path.join(self.zmd.result_dir, m_name)
        cv.imwrite(map_hl_path, self.zmd.map_planning)
        # save map segmentation
        m_name_s = "map_segmentation.png"
        map_s_path = os.path.join(self.zmd.result_dir, m_name_s)
        cv.imwrite(map_s_path, self.zmd.map_segmentation)

    def save_map_cleaning_data(self):
        """ save map showing cleaned area and uncleaned area. """
        if not self.dcfg.save_map_planning:
            return
        m_name = "map_cleaned.png"
        map_hl_path = os.path.join(self.zmd.result_dir, m_name)
        cv.imwrite(map_hl_path, self.zmd.map_clean_data)
