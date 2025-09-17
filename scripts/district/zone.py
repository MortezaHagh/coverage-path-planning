import os
import json
import cv2 as cv
import numpy as np
from scripts.base.logger import MyLogger
from scripts.segmentation.cell_base import CellBase
from scripts.district.zone_map_data import ZoneMapData
from scripts.base.configs import DistrictConfigs, PlanningConfigs
from scripts.segmentation.segmentation import SegmentationInterface
from scripts.planning.coverage_path_plan_handler import CoveragePathPlanHandler
from scripts.sequence.sequence import Sequence


class ZoneStats:
    """Zone statistics class to store zone statistics. 
    Attributes:
        cleaning_area_px (int): area to be cleaned in pixels
        cleaning_area_m (float): area to be cleaned in square meters
        cleaned_area_px (int): cleaned area in pixels
        cleaned_area_m (float): cleaned area in square meters
        uncleaned_area_m (float): uncleaned area in square meters
        overlaped_area_m (float): overlapped area in square meters
        coverage_p (float): coverage percentage
        overlap_p (float): overlap percentage
    """

    def __init__(self):
        self.cleaning_area_px = None
        self.cleaning_area_m = None
        self.cleaned_area_px = 0
        self.cleaned_area_m = 0
        self.uncleaned_area_m = None
        self.overlaped_area_m = None
        self.coverage_p = None
        self.overlap_p = None

    def get_data_dict(self) -> dict:
        """Get statistics data as a dictionary. """
        return {
            "cleaning_area_px": int(self.cleaning_area_px),
            "cleaning_area_m": float(self.cleaning_area_m),
            "cleaned_area_px": int(self.cleaned_area_px),
            "cleaned_area_m": float(self.cleaned_area_m),
            "uncleaned_area_m": float(self.uncleaned_area_m),
            "overlaped_area_m": float(self.overlaped_area_m),
            "coverage_p": float(self.coverage_p),
            "overlap_p": float(self.overlap_p)
        }

    def print_stats(self):
        print("[Results]: ******************************************************* ")
        print(f"[Results]: cleaning area: {self.cleaning_area_m:.2f} m^2")
        print(f"[Results]: cleaned area: {self.cleaned_area_m:.2f} m^2")


class ZoneData:
    """ Zone Data class to store zone map data and run coverage planning.  

    Attributes:
        ind (int): zone index
        zmd (ZoneMapData): zone map data
        segmentation_interface (SegmentationInterface): zone segmentation interface
        cells (list[CellBase]): list of cells from segmentation
        cov_path_planner (CoveragePathPlanner): coverage path planner
        complete_path (np.ndarray): complete path (todo)
        st (ZoneStats): zone statistics
    """

    def __init__(self, ind: int, zmd: ZoneMapData):
        self.ns: str = f"ZoneData-{ind}"
        self.lg = MyLogger(self.ns)

        # attributes
        self.ind: int = ind
        self.zmd: ZoneMapData = zmd

        # configs
        self.dcfg: DistrictConfigs = DistrictConfigs()
        self.pcfgs: PlanningConfigs = PlanningConfigs()

        # # # segmentation
        self.segmentation_interface: SegmentationInterface = SegmentationInterface(self.zmd)

        # # # coverage path planner
        self.cov_path_planner: CoveragePathPlanHandler = CoveragePathPlanHandler(self.zmd)

        # sequence
        self.sequence: Sequence = Sequence(self.zmd)

        # resulting data
        self.cells: list[CellBase] = []
        self.complete_path: np.ndarray = None  # todo

        # statistics
        self.st: ZoneStats = ZoneStats()

        # initialize
        self.initialize()

    def initialize(self):
        """ - initialize zone data and create map planning.
            - create segmentor and coverage path planner.
        """
        self.zmd.save_data()
        self.create_map_planning()

    def coverage_planning(self):
        """ run coverage path planning for the zone and then post planning. """

        # ************
        self.lg.info1(f"Step: running segmentation using method '{self.dcfg.segmentation_method}'")
        self.segmentation_interface.run()
        self.cells = self.segmentation_interface.cells

        # ************
        self.lg.info1(f"Step: running coverage path planning using method '{self.dcfg.covpp_method}'")
        self.cov_path_planner.plan(cells=self.cells)

        # ************
        self.lg.info1(f"Step: creating sequence and full path using method '{self.dcfg.sequence_method}'")
        self.sequence.run(cells=self.cells, map_polygons=self.segmentation_interface.map_polygons)

        # ************
        self.lg.info1(f"Step: zone post planning")
        self.post_planning()

    def post_planning(self):
        """ post planning after coverage path planning. 
        calculating stats and updating map_planning and map_clean_data. 
        """
        self.calculate_cleaning_area()
        self.add_cleaned_area_to_map()
        self.calculate_cleaned_area()
        self.save_map_planning()
        self.save_map_cleaning_data()
        self.save_result()
        self.st.print_stats()

    # ==================================================================================================================

    def calculate_cleaning_area(self):
        self.st.cleaning_area_px = np.sum(self.zmd.map_cleaning == 255)
        self.st.cleaning_area_m = round(self.st.cleaning_area_px * (self.zmd.resolution**2), 2)

    def calculate_cleaned_area(self):
        paths_len = self.sequence.st.len_m
        cleaning_r = self.pcfgs.cleaning_r
        area = paths_len * (2 * cleaning_r)
        # overlaped_area_m
        self.st.overlaped_area_m = round(area - self.st.cleaned_area_m, 2)
        self.st.overlap_p = round(self.st.overlaped_area_m / self.st.cleaning_area_m * 100, 2)
        # uncleaned_area_m
        self.st.uncleaned_area_m = round(self.st.cleaning_area_m - self.st.cleaned_area_m, 2)
        # coverage percentage
        self.st.coverage_p = round(self.st.cleaned_area_m / self.st.cleaning_area_m * 100, 2)

    def add_cleaned_area_to_map(self):
        """ add cleaned area to map_clean_data and update map_planning with cleaning paths. """

        # map_cleaning: from zone map data (without any paths)
        # used for creating map_clean_data
        map_cleaning = self.zmd.map_cleaning.copy()
        f_path = self.sequence.f_path_px
        cleaning_r_px = int(self.pcfgs.cleaning_r / self.zmd.resolution)

        if f_path is None:
            self.lg.error("fill path is None. Cannot add cleaned area to map.")
            self.zmd.map_clean_data = map_cleaning.copy()
            return

        # Draw cleaned paths on a mask
        cleaned_acc = np.zeros_like(map_cleaning, dtype=np.uint8)
        hw_bd = self.zmd.hw_bd_px
        h, w = hw_bd["h_min"], hw_bd["w_min"]
        color_p = 1
        # for path in paths:
        for i in range(f_path.n - 1):
            segment_mask = np.zeros_like(map_cleaning, dtype=np.uint8)
            pt1 = tuple(map(int, [f_path.point(i).x-w, f_path.point(i).y-h]))
            pt2 = tuple(map(int, [f_path.point(i+1).x-w, f_path.point(i+1).y-h]))
            cv.line(segment_mask, pt1, pt2, color=color_p, thickness=2 * cleaning_r_px)
            cleaned_acc = cv.add(cleaned_acc, segment_mask)
        max_overlap = np.max(cleaned_acc)

        # map_clean_data: show uncleaned area and overlaps
        d_c = int(np.floor(250/max_overlap))
        map_clean_data = map_cleaning.copy()
        map_clean_data = cv.cvtColor(map_clean_data, cv.COLOR_GRAY2BGR)
        n, m = map_clean_data.shape[:2]
        for i in range(n):
            for j in range(m):
                if map_cleaning[i, j] == 255:
                    if cleaned_acc[i, j] == 0:
                        map_clean_data[i, j] = (0, 0, 200)  # bgr
                    else:
                        map_clean_data[i, j] = (0, d_c*cleaned_acc[i, j], 0)
        # for path in paths:
        for p in f_path.points[1:-1]:
            pz = (int(p.x-w), int(p.y-h))
            map_clean_data = cv.circle(map_clean_data, pz, cleaning_r_px, (0, d_c, 0), -1)
        self.zmd.map_clean_data = map_clean_data

        # update map planning with cleaning paths
        add_cleaned_area_to_map_planning = False
        if add_cleaned_area_to_map_planning:
            c = 40
            color = (c, c, c)
            map_p_c = self.zmd.map_planning.copy()
            # for path in paths:
            for i in range(f_path.n - 1):
                pt1 = tuple(map(int, f_path.point(i).xy))
                pt2 = tuple(map(int, f_path.point(i+1).xy))
                cv.line(map_p_c, pt1, pt2, color=color, thickness=2 * cleaning_r_px)
            alpha = 0.6
            self.zmd.map_planning = cv.addWeighted(self.zmd.map_planning, alpha, map_p_c, 1 - alpha, 0)

        # Only count cleaned area where there was free space (white in map_cleaned)
        # cleaned_area_px = np.sum((cleaned_acc > 0) & (map_cleaned == 255))
        cleaned_area_px = np.sum(cleaned_acc > 0)
        cleaned_area_m = cleaned_area_px * (self.zmd.resolution ** 2)
        self.st.cleaned_area_px = cleaned_area_px
        self.st.cleaned_area_m = round(cleaned_area_m, 2)

    # ==================================================================================================================

    def create_map_planning(self):
        """ create initial map_planning with zone boundary. """
        map_planning = self.zmd.map_planning
        vertices = self.zmd.vertices
        map_planning = cv.polylines(map_planning, [vertices], isClosed=True, color=(100, 0, 150), thickness=2)
        self.zmd.map_planning = map_planning

    def save_map_planning(self):
        if not self.dcfg.save_map_planning:
            return
        m_name = "map_planning.png"
        map_hl_path = os.path.join(self.zmd.result_path, m_name)
        cv.imwrite(map_hl_path, self.zmd.map_planning)
        # self.lg.info(f"{self.zmd.ind}][save]: map planning saved to: {m_name}")

    def save_map_cleaning_data(self):
        """ save map showing cleaned area and uncleaned area. """
        if not self.dcfg.save_map_planning:
            return
        m_name = "map_cleaned.png"
        map_hl_path = os.path.join(self.zmd.result_path, m_name)
        cv.imwrite(map_hl_path, self.zmd.map_clean_data)
        # self.lg.info(f"{self.zmd.ind}][save]: map cleaning saved to: {m_name}")

    # def point_m_to_px(self, p: tuple[float, float]) -> tuple[int, int]:
    #     res = self.zmd.resolution
    #     return (int(p[0] / res), int(p[1] / res))

    def save_result(self):
        """ save zone statistics and planner statistics of the zone to json file. """
        # save zone statistics to json file
        zone_stats = self.st.get_data_dict()
        planner_stats = self.sequence.st.get_data_dict()
        result = {
            "map": self.zmd.orig_map_data.name,
            "zone_ind": self.zmd.ind,
            "method": self.dcfg.method,
            "zone_stats": zone_stats,
            "planner_stats": planner_stats
        }
        # rewrite result file
        with open(self.zmd.result_file, 'wt') as f:
            json.dump(result, f, indent=4)
