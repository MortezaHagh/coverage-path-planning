"""Processed cell class with advanced geometric analysis for coverage planning."""
import cv2 as cv
import numpy as np
from shapely import Polygon
from covpp.gui.rectangle import Rectangle
from covpp.base.logger import MyLogger
from covpp.base.configs import PipeLineConfig
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.eval_rectangles import EvalRectangles
from covpp.district.zone_map_data import ZoneMapData
# from shapely import maximum_inscribed_circle, minimum_bounding_circle


class PCell(CellBase):
    """Enhanced cell with geometric analysis for optimal coverage planning.

    Extends the base Cell class to include:
    - Binary map representation of the cell area
    - Rectangle-like property evaluation (circumscribed and inscribed)
    - Main direction extraction using Hough Transform.
    """

    def __init__(self, cell: CellBase, zmd: ZoneMapData) -> None:
        super().__init__(cell.id, cell.poly)
        self.lg = MyLogger(f"PCell-{self.id}")

        # adjacent ids
        self.adjacent_ids = cell.adjacent_ids

        # zone map data
        self.zmd: ZoneMapData = zmd

        # cell map
        # map boundary is either original cell polygon or inscribed rectangle.
        self.map: np.ndarray = None                 # map of the cell area (free = 255, occupied = 0)
        self.map_xy_bd: dict = {}                   # cell local map bounds
        self.global_xy = np.array([0, 0], dtype=np.int32)    # global map offset (x_offset, y_offset)

        # rectangle-like properties
        self.rect_like: bool = False              # whether the cell is rectangle-like or inscribed rectangle-like
        # inscribed rectangle polygon (expanded)
        # also it is rotated to start from closest point to the start of the original polygon
        self.rectangle_poly: Polygon = None

        # # major direction defined in base cell
        # self.major_direction: float = 0.0

        # ----------------------------------------------------------------------

        # find rectangle-like properties
        if PipeLineConfig.use_insc_ret:
            self.eval_cell_rectangles()

        # create cell map
        self.create_cell_map(self.zmd.map_ws.copy(), zmd.rectangle)

        # find main directions
        self.find_cell_main_direction()

    def eval_cell_rectangles(self) -> None:
        """Evaluate cell for rectangle-like properties to optimize planning. """

        er = EvalRectangles(self.id, self.poly, self.robot_r_px)
        self.rect_like = er.rect_like
        self.rectangle_poly = er.rectangle_poly

    def create_cell_map(self, zone_map: np.ndarray, zone_rectangle: Rectangle) -> None:
        """ Create binary map for the given cell. """

        # determine bounding box  of the cell in the zone map
        x_min = int(self.x_min_p - zone_rectangle.x_min)
        y_min = int(self.y_min_p - zone_rectangle.y_min)
        x_max = int(self.x_max_p - zone_rectangle.x_min)
        y_max = int(self.y_max_p - zone_rectangle.y_min)
        n = 5
        x_min = max(x_min-n, 0)
        y_min = max(y_min-n, 0)
        x_max = min(x_max+n, zone_rectangle.x_max)
        y_max = min(y_max+n, zone_rectangle.y_max)
        self.global_xy = np.array([x_min + zone_rectangle.x_min, y_min + zone_rectangle.y_min], dtype=np.int32)

        # crop map_ws to bounding box
        map_ws = zone_map.copy()
        map_ws = map_ws[y_min: y_max, x_min: x_max]

        # create empty map
        map_cell = np.zeros_like(map_ws, dtype=np.uint8)

        # get polygon points of the cell
        points = self.points
        if self.rect_like:
            points = np.array(self.rectangle_poly.exterior.coords, dtype=np.int32)

        # points in the new local map coordinates
        poly_pts = points - self.global_xy

        # fill cell polygon area with free space (255)
        cv.fillPoly(map_cell, [poly_pts], 255)

        # mask with free space in the original map
        map_cell = cv.bitwise_and(map_cell, map_ws)

        # # for debugging
        # map_viz = cv.cvtColor(map_cell, cv.COLOR_GRAY2BGR)
        # # add poly_pts to map_viz by red color
        # cv.polylines(map_viz, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        # cv.namedWindow(f"Cell {self.id} Map", cv.WINDOW_NORMAL)
        # cv.imshow(f"Cell {self.id} Map", map_viz)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        self.map = map_cell

    def find_cell_main_direction(self) -> None:
        """ Find main directions of each cell using Hough Transform.
            to be implemented in child classes.
        """
        if self.rect_like:
            self.major_direction = self.main_direction_by_cell_rectangle()
        else:
            self.major_direction = self.cell_main_direction()

    # ==================================================================================================================
    # main direction of the cell
    # ==================================================================================================================

    def main_direction_by_cell_rectangle(self) -> float:
        """ Get main direction from cell rectangle orientation. """

        # Extract coordinates (last point == first point, so we skip last)
        rect = self.rectangle_poly
        coords = list(rect.exterior.coords)

        longest_length = 0
        longest_angle = None

        for i, (x1, y1) in enumerate(coords):
            x2, y2 = coords[(i+1) % len(coords)]
            dx, dy = (x2 - x1, y2 - y1)
            length = np.hypot(dx, dy)

            if length > longest_length:
                longest_length = length
                # Angle in radians relative to +X axis
                longest_angle = np.arctan2(dy, dx)
        return (longest_angle + np.pi/2) % np.pi

    def cell_main_direction(self) -> float:
        """ Find main direction of the cell using Hough Transform. """

        # Edge detection
        edges = cv.Canny(self.map, 50, 150, apertureSize=3)

        # Hough Transform # tweak threshold if needed
        threshold = 120   # start threshold
        min_threshold = 50  # don't go below this
        lines = None
        while threshold >= min_threshold and lines is None:
            lines = cv.HoughLines(edges, 1, np.pi/180, threshold)
            if lines is None:
                threshold -= 20  # decrease threshold step-by-step

        # If no lines detected, fallback to default angle (e.g., 0)
        angles = []
        if lines is not None and len(lines) > 0:
            # rho, theta
            for _, theta in lines[:, 0]:
                angles.append(theta)
        else:
            # Fallback: assume horizontal direction
            angles.append(0.0)

        # select dominant wall orientation (not median)
        dominant_angle = max(set(angles), key=angles.count)
        major_direction = (dominant_angle) % np.pi

        # # visualization
        # # Draw arrow from center of map in free-space direction
        # h, w = self.map.shape
        # center = (w//2, h//2)
        # length = min(w, h) // 3  # arrow length

        # dx = int(length * np.cos(major_direction))
        # dy = int(length * np.sin(major_direction))

        # # Convert grayscale map to BGR for drawing
        # vis = cv.cvtColor(self.map, cv.COLOR_GRAY2BGR)
        # # add lines to vis by green color
        # if lines is not None:
        #     for rho, theta in lines[:, 0]:
        #         a = np.cos(theta)
        #         b = np.sin(theta)
        #         x0 = a * rho
        #         y0 = b * rho
        #         pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        #         pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        #         cv.line(vis, pt1, pt2, (0, 255, 0), 2)
        # # draw major direction arrow
        # cv.arrowedLine(vis, center, (center[0] + dx, center[1] + dy), (0, 0, 255), 3, tipLength=0.2)
        # # display image, lines, and major direction
        # cv.imshow(f"Cell {self.id} Main Direction", vis)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return major_direction
