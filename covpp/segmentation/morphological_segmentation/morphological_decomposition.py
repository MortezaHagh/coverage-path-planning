

import cv2 as cv
import numpy as np
from shapely import Polygon
from scipy import ndimage
from collections import deque
from matplotlib import pyplot as plt
from covpp.base.logger import MyLogger
from covpp.district.zone_map_data import ZoneMapData

OCCUPIED = 0
FREE = 255


class MorphologicalSegmentation:

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("MorphologicalSegmentation")
        self.lg.info2("Initializing Morphological Segmentation.")

        # data
        self.zmd = zmd

        # parameters
        erosion_kernel_size = 3
        self.kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)

        # max min area based on robot size
        robot_radius_px = zmd.robot_radius_px
        robot_area_px = int(np.pi * (robot_radius_px ** 2))
        self.min_area = robot_area_px * 5
        self.max_area_factor = 4

        # attributes
        self.polygons: list[Polygon] = []                   # list of polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray      # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray      # map_ws with segmentation lines

    def run(self) -> None:
        labels_map = self.morphological_decomposition()
        labels_map = self.wavefront_propagation(labels_map=labels_map)
        self.create_segmentation_map_polygons(labels_map)
        # self.plot()

    def morphological_decomposition(self) -> np.ndarray:

        free_map = self.zmd.map_ws.astype(np.uint8)
        labels_map = np.zeros(free_map.shape, np.int32)

        # Distance to the closest obstacle gives the erosion depth of each pixel.
        erosion_depth = cv.distanceTransform(free_map, cv.DIST_C, 3)
        remaining_depth = erosion_depth.copy()

        room_id = 1

        max_depth = int(np.floor(erosion_depth.max()))
        connectivity_8 = np.ones((3, 3), dtype=np.int8)

        for erosion_step in range(1, max_depth + 1):

            if not np.any(remaining_depth >= erosion_step):
                continue

            # Thresholding the distance transform reproduces the erosion layers
            # without repeated morphological operations.
            cc, num = ndimage.label(remaining_depth >= erosion_step, structure=connectivity_8)

            max_area = (self.max_area_factor * erosion_step * self.min_area)

            for component_id in range(1, num + 1):

                mask = cc == component_id
                area = np.count_nonzero(mask)

                if (area >= self.min_area and area <= max_area):

                    labels_map[mask] = room_id
                    room_id += 1

                    # Remove accepted components so they are not reconsidered at shallower depths.
                    remaining_depth[mask] = 0

        return labels_map

    def wavefront_propagation(self, labels_map: np.ndarray) -> np.ndarray:

        binary_map = self.zmd.map_ws.astype(np.uint8)

        queue = deque()
        rows, cols = binary_map.shape

        n_labels = np.max(labels_map) - 1  # -1 because labels start from 1
        adjacency_matrix = np.zeros((n_labels + 1, n_labels + 1), dtype=np.int32)

        # Initialize queue
        ys, xs = np.where(labels_map > 0)
        for y, x in zip(ys, xs):
            queue.append((y, x))

        # BFS
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:

            y, x = queue.popleft()

            current_label = labels_map[y, x]

            for dy, dx in directions:

                ny = y + dy
                nx = x + dx

                if (
                    ny < 0
                    or nx < 0
                    or ny >= rows
                    or nx >= cols
                ):
                    continue

                if binary_map[ny, nx] == OCCUPIED:
                    continue

                if labels_map[ny, nx] != 0:  # already labeled
                    if labels_map[ny, nx] != current_label:
                        adjacency_matrix[current_label - 1, labels_map[ny, nx] - 1] = 1
                        adjacency_matrix[labels_map[ny, nx] - 1, current_label - 1] = 1
                    continue

                labels_map[ny, nx] = current_label

                queue.append((ny, nx))

        self.adjacency_matrix = adjacency_matrix
        return labels_map

    def create_segmentation_map_polygons(self, labels_map: np.ndarray) -> None:
        polygons = []
        map_ws_segmented = self.zmd.map_ws.copy()
        for label in range(1, np.max(labels_map) + 1):
            mask = labels_map == label
            contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(map_ws_segmented, contours, -1, OCCUPIED, thickness=2)
            # create shapely polygon for biggest contour
            if len(contours) == 1:
                pts = contours[0].reshape(-1, 2) + self.zmd.global_xy
                polygon = Polygon(pts)
                polygons.append(polygon)
            elif len(contours) > 1:
                largest_contour = max(contours, key=cv.contourArea)
                pts = largest_contour.reshape(-1, 2) + self.zmd.global_xy
                polygon = Polygon(pts)
                polygons.append(polygon)
                self.lg.warn(f"Multiple contours found for label {label}. Using the largest contour.")
            else:
                self.lg.warn(f"No contours found for label {label}.")
                raise ValueError(f"No contours found for label {label}.")

        self.polygons = polygons
        self.map_ws_segmented = map_ws_segmented

    @staticmethod
    def viz_img(img: np.ndarray, name: str):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def plot(self):

        # plot polygons
        colors = plt.cm.get_cmap('tab20', len(self.polygons))
        fig, ax = plt.subplots(figsize=(8, 8))
        for i, polygon in enumerate(self.polygons):
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=colors(i), ec='black')
            # id at center
            ax.text(polygon.centroid.x, polygon.centroid.y, str(self.polygons.index(polygon)), fontsize=12, ha='center')

        # show adjacency polygons by line
        for i, polygon in enumerate(self.polygons):
            for j, adjacent in enumerate(self.adjacency_matrix[i]):
                if adjacent:
                    x1, y1 = polygon.centroid.xy
                    x2, y2 = self.polygons[j].centroid.xy
                    ax.plot([x1[0], x2[0]], [y1[0], y2[0]], 'b--', linewidth=1)
        ax.set_title("Morphological Segmentation Polygons")
        ax.set_aspect('equal')
        plt.show()
