import cv2 as cv
import numpy as np
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

        # configs
        self.dcfgs = DistrictConfigs()

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

    def run(self) -> None:
        """ Run segmentation.
        """

        # segmentation method
        segmentation_method = self.dcfgs.segmentation_method

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

        #
        self.find_cells_main_directions()

    def find_cells_main_directions(self) -> None:
        """ Find main directions of each cell using Hough Transform.
            to be implemented in child classes.
        """
        for cell in self.cells:
            if cell.rect_like:
                cell.free_angle = self.main_direction_by_cell_rectangle(cell)
            else:
                map_cell = self.create_cell_map(cell)
                cell.free_angle = self.cell_main_direction(map_cell)

    def main_direction_by_cell_rectangle(self, cell: CellBase) -> float:
        """ Get main direction from cell rectangle orientation. """

        # Extract coordinates (last point == first point, so we skip last)
        rect = cell.rectangle_poly
        coords = list(rect.exterior.coords)[:-1]

        longest_length = 0
        longest_angle = None

        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i+1) % len(coords)]

            dx, dy = (x2 - x1, y2 - y1)
            length = np.hypot(dx, dy)

            if length > longest_length:
                longest_length = length
                # Angle in radians relative to +X axis
                longest_angle = np.arctan2(dy, dx)
        return (longest_angle + np.pi/2) % np.pi

    def create_cell_map(self, cell: CellBase) -> np.ndarray:
        """ Create binary map for the given cell. """

        # get polygon bounding box
        minx, miny, maxx, maxy = cell.poly.bounds
        minx = minx - self.zmd.rectangle.x_min
        miny = miny - self.zmd.rectangle.y_min

        # crop map_ws to bounding box
        map_ws = self.zmd.map_ws.copy()
        map_ws = map_ws[int(miny):int(maxy), int(minx):int(maxx)]

        # create empty map
        map_cell = np.zeros_like(map_ws, dtype=np.uint8)

        # get polygon points of the cell
        points = cell.points
        if self.dcfgs.use_insc_ret and cell.rect_like:
            points = np.array(cell.rectangle_poly.exterior.coords, dtype=np.int32)

        # fill cell polygon area with free space (255)
        # poly_pts = points - self.zmd.rectangle.min_point
        poly_pts = points - np.array([[minx, miny]], dtype=np.int32)
        cv.fillPoly(map_cell, [poly_pts], 255)

        # mask with free space in the original map
        map_cell = cv.bitwise_and(map_cell, map_ws)

        return map_cell

    def cell_main_direction(self, map_cell: np.ndarray) -> float:
        """ create MapPolygons object for the cell area in the map"""

        # Edge detection
        edges = cv.Canny(map_cell, 50, 150, apertureSize=3)

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
            for rho, theta in lines[:, 0]:
                angles.append(theta)
        else:
            # Fallback: assume horizontal direction
            angles.append(0.0)

        # Compute dominant wall orientation
        dominant_angle = np.median(angles)   # robust estimate
        # free_angle = (dominant_angle + np.pi/2) % np.pi
        free_angle = (dominant_angle) % np.pi

        # Convert grayscale map to BGR for drawing
        vis = cv.cvtColor(map_cell, cv.COLOR_GRAY2BGR)

        # # Draw arrow from center of map in free-space direction
        # h, w = map_cell.shape
        # center = (w//2, h//2)
        # length = min(w, h) // 3  # arrow length

        # dx = int(length * np.cos(free_angle))
        # dy = int(length * np.sin(free_angle))

        # cv.arrowedLine(vis, center, (center[0] + dx, center[1] + dy), (0, 0, 255), 3, tipLength=0.2)

        # # Show results
        # cv.imshow("Map with Direction", vis)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return free_angle
