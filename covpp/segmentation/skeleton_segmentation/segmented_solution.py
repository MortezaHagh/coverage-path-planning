
import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
from covpp.base.logger import MyLogger
from covpp.segmentation.skeleton_segmentation.data_structures import Configs, CriticalPoint

OCCUPIED = 0
FREE = 255


class CreateSegmentationSolution:
    def __init__(self, map_ws: np.ndarray, map_ws_segmented: np.ndarray, robot_radius_px: int, origin_coord: tuple[int, int]):
        self.lg = MyLogger(name="CreateSegmentationSolution")
        self.map_ws: np.ndarray = map_ws
        self.map_ws_segmented: np.ndarray = map_ws_segmented
        self.robot_radius_px: int = robot_radius_px
        self.origin_coord: tuple[int, int] = origin_coord

        # attributes
        self.segmented_map: np.ndarray = None
        self.seg_ids: list[int] = None
        self.local_polygons: list[Polygon] = None
        self.polygons: list[Polygon] = None
        self.adjacency_matrix: np.ndarray = None
        self.error_point_segments: list = []

    def run_by_critical_points(self, base_pts: list[list[tuple[int, int]]], critical_pts: list[CriticalPoint]):
        self.create_segmented_map(base_pts)
        self.create_adjacency_matrix_by_critical_points(critical_pts)
        self.create_polygons()

    def run_by_map(self, base_pts: list[list[tuple[int, int]]]):
        self.create_segmented_map(base_pts)
        self.create_adjacency_matrix_by_segmented_map()
        self.create_polygons()

    # ==================================================================================================================
    # segmentation map, polygons and adjacency matrix creation
    # ==================================================================================================================

    def create_segmented_map(self, base_pts: list[list[tuple[int, int]]]) -> None:
        """ Create segmented map based on base points.

        Steps:
        - Connect each two base points with a line.
        - Find contours and fill areas to create segmented map.
        - Each area will have a unique gray value. (later used in creating cells)
        """

        seg_map = self.map_ws_segmented.copy()

        # create a line connecting each two base points consecutively
        for bases in base_pts:
            n_bases = len(bases)
            for i in range(n_bases):
                j = (i + 1) % n_bases
                base1 = bases[i]
                base2 = bases[j]
                cv.line(seg_map, (base1[1], base1[0]), (base2[1], base2[0]), OCCUPIED, thickness=2)

        # create map_ws_segmented ----------------------------------------------
        self.map_ws_segmented = seg_map.copy()

        seg_ids = []

        # find contours and fill areas
        contours, hierarchy = cv.findContours(seg_map, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        self.lg.debug(f"Found {len(contours)} contours in segmented map.")

        # filter contours: remove those with children and small areas
        filtered_contours = []
        it = 0
        for c, h in zip(contours, hierarchy):

            # skip contours with children
            if not self.is_valid_contour(h, hierarchy):
                continue

            # minimum area threshold
            area = cv.contourArea(c)
            if area >= Configs.min_area_thresh_for_contour:
                filtered_contours.append(c)
                it += 1

        # check too many contours
        self.lg.debug(f"Found {len(filtered_contours)} filtered contours in segmented map.")
        if len(filtered_contours) > 254:
            self.lg.warn(f"Too many contours found in segmented map: {len(filtered_contours)}.")

        # fill area with different gray value (used for segment IDs)
        for i, cnt in enumerate(filtered_contours):
            new_id = i + 1
            seg_ids.append(new_id)
            cv.fillPoly(seg_map, pts=[cnt], color=new_id)
            cv.drawContours(seg_map, [cnt], 0, FREE, thickness=1)

        # # debugging
        # self.viz_img(seg_map, "create_segmented_map")

        if not seg_ids:
            raise ValueError("No segments found in segmented map.")

        self.seg_map = seg_map
        self.seg_ids = seg_ids
        self.local_polygons = filtered_contours

    def create_adjacency_matrix_by_critical_points(self, critical_pts: list[CriticalPoint]) -> None:
        """ Create adjacency matrix between segments based on critical points.

        Method:
        - For each critical point, find the segments around it by checking a small canvas.
        - Mark those segments as adjacent in the adjacency matrix.
        """

        adjacency_matrix = np.zeros((len(self.seg_ids), len(self.seg_ids)), dtype=np.bool)
        h, w = self.seg_map.shape
        # canvas_size = 40 // 2
        canvas_size = 4 * self.robot_radius_px  # param

        for cp in critical_pts:
            y, x = cp.coord
            y_min = max(0, y - canvas_size)
            y_max = min(h, y + canvas_size)
            x_min = max(0, x - canvas_size)
            x_max = min(w, x + canvas_size)
            canvas = self.seg_map[y_min:y_max, x_min:x_max]
            unique_ids = np.unique(canvas)
            # remove OCCUPIED and FREE
            unique_ids = unique_ids[(unique_ids != OCCUPIED) & (unique_ids != FREE)]
            n_ids = len(unique_ids)
            if n_ids < 2:
                self.error_point_segments.append(cp.coord)
                self.lg.warn(f"Less than 2 segments found near critical point at {cp.coord}.")

            for i in range(n_ids):
                for j in range(i + 1, n_ids):
                    adjacency_matrix[unique_ids[i] - 1, unique_ids[j] - 1] = True
                    adjacency_matrix[unique_ids[j] - 1, unique_ids[i] - 1] = True
        self.adjacency_matrix = adjacency_matrix

    def create_adjacency_matrix_by_segmented_map(self) -> None:
        # go through all contour (local polygon) points
        # for each point, check a canvas around it
        # if there are more than 1 unique segment IDs, mark them as adjacent in the adjacency matrix
        adjacency_matrix = np.zeros((len(self.seg_ids), len(self.seg_ids)), dtype=np.bool)
        h, w = self.seg_map.shape
        # canvas_size = 40 // 2
        canvas_size = 4 * self.robot_radius_px  # param

        # for each local polygon (contour)
        for cnt in self.local_polygons:
            for pt in cnt:
                y, x = pt[0][1], pt[0][0]
                y_min = max(0, y - canvas_size)
                y_max = min(h, y + canvas_size)
                x_min = max(0, x - canvas_size)
                x_max = min(w, x + canvas_size)
                canvas = self.seg_map[y_min:y_max, x_min:x_max]
                unique_ids = np.unique(canvas)
                # remove OCCUPIED and FREE
                unique_ids = unique_ids[(unique_ids != OCCUPIED) & (unique_ids != FREE)]
                n_ids = len(unique_ids)
                if n_ids < 2:
                    continue
                for i in range(n_ids):
                    for j in range(i + 1, n_ids):
                        adjacency_matrix[unique_ids[i] - 1, unique_ids[j] - 1] = True
                        adjacency_matrix[unique_ids[j] - 1, unique_ids[i] - 1] = True
        self.adjacency_matrix = adjacency_matrix

    def create_polygons(self) -> None:

        # create shapely polygons for each contour
        from shapely.geometry import Polygon
        polygons = []
        for i, cnt in enumerate(self.local_polygons):
            pts = cnt.reshape(-1, 2)
            if len(pts) < 3:
                self.lg.error(f"Contour {i} has less than 3 points, cannot create polygon.")
                continue
            # add global coordinate conversion
            g_pts = self.origin_coord + pts
            poly = Polygon(g_pts)
            polygons.append(poly)
        self.polygons = polygons

    # ==================================================================================================================
    # Utility methods
    # ==================================================================================================================

    @staticmethod
    def is_valid_contour(h, hierarchy) -> bool:
        """ Check if contour is valid based on its hierarchy. A valid contour should not have both parent and children. """

        def find_level(h, hierarchy):
            """ Find the level of contour in hierarchy. """
            level = 0
            parent = h[3]
            while parent != -1:
                level += 1
                parent = hierarchy[parent][3]
            return level

        # if level is odd, it is a hole contour
        level = find_level(h, hierarchy)
        if level % 2 == 1:
            return False
        return True
        # # Check if contour has both parent and children in hierarchy.
        # return not(h[3] != -1 and h[2] != -1)
