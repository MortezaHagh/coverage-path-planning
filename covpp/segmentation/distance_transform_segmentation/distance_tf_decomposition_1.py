

import cv2 as cv
import numpy as np
from shapely import Polygon
from scipy import ndimage
from collections import deque
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from covpp.base.logger import MyLogger
from covpp.district.zone_map_data import ZoneMapData

OCCUPIED = 0
FREE = 255


class DistanceTransformSegmentation:

    def __init__(self, zmd: ZoneMapData):
        self.lg = MyLogger("DistanceTransformSegmentation")
        self.lg.info2("Initializing Distance Transform Segmentation.")

        # data
        self.zmd = zmd

        # parameters
        robot_radius_px = zmd.robot_radius_px
        robot_area_px = int(np.pi * (robot_radius_px ** 2))
        self.min_room_area: int = robot_area_px * 5

        self.sigma = 0.5 * robot_radius_px                  # Gaussian smoothing applied to the distance transform to suppress noise and merge nearby room-center peaks.
        self.min_peak_distance = 8 * robot_radius_px        # Minimum allowed distance (pixels) between detected room-center peaks to avoid multiple seeds per room.
        self.min_peak_value = 2.2 * robot_radius_px         # Minimum distance-to-wall (pixels) for a local maximum to be considered a valid room center, filtering out peaks in narrow passages.

        # attributes
        self.polygons: list[Polygon] = []                   # list of polygons for each segment
        self.adjacency_matrix: np.ndarray = np.ndarray      # adjacency matrix between segments
        self.map_ws_segmented: np.ndarray = np.ndarray      # map_ws with segmentation lines

    def run(self) -> None:
        labels_map, distance = self.distance_tf_decomposition()
        self.find_adjacency_matrix(labels_map, distance)
        self.create_segmentation_map_polygons(labels_map)
        # self.plot()

    def distance_tf_decomposition(self) -> tuple[np.ndarray, np.ndarray]:

        binary_map = self.zmd.map_ws.astype(np.uint8)
        distance = cv.distanceTransform(binary_map, cv.DIST_L2, 5)

        # Smooth
        distance = ndimage.gaussian_filter(distance, self.sigma)

        ####################################################
        # Detect room centers
        ####################################################

        peaks = peak_local_max(
            distance,
            min_distance=self.min_peak_distance,
            threshold_abs=self.min_peak_value,
            labels=binary_map,
        )

        # Create markers
        markers = np.zeros_like(binary_map, dtype=np.int32)
        for i, (r, c) in enumerate(peaks):
            markers[r, c] = i + 1

        # Dilate markers to improve watershed segmentation, especially in narrow passages
        markers = ndimage.maximum_filter(markers, size=3)

        markers = label(markers)

        ####################################################
        # Watershed
        ####################################################

        labels = watershed(-distance, markers, mask=binary_map,)

        # Merge tiny regions
        labels = self.merge_small_regions(labels)

        # # debug: visualize the segmentation result
        # self.viz_img(labels.astype(np.uint8) * 20, "Distance Transform Segmentation Labels")

        return labels, distance

    def merge_small_regions(self, labels):

        ids = np.unique(labels)

        ids = ids[ids != 0]

        for rid in ids:

            region = labels == rid

            area = np.count_nonzero(region)

            if area >= self.min_room_area:
                continue

            dilated = ndimage.binary_dilation(region)

            neighbors = labels[dilated]

            neighbors = neighbors[
                (neighbors != rid) &
                (neighbors != 0)
            ]

            if len(neighbors) == 0:
                continue

            values, counts = np.unique(
                neighbors,
                return_counts=True,
            )

            best = values[np.argmax(counts)]

            labels[region] = best

        # fix labels to be consecutive integers starting from 1
        unique_labels = np.unique(labels)
        new_labels = np.zeros_like(labels)
        for new_id, old_id in enumerate(unique_labels):
            if old_id == 0:
                continue
            new_labels[labels == old_id] = new_id

        return new_labels

    def find_adjacency_matrix(self, labels_map: np.ndarray, distance: np.ndarray) -> np.ndarray:
        # iterate over pixels in labels map with distance equal to 1 (i.e., boundary pixels) and check if they have neighbors with different labels
        n_labels = np.max(labels_map) - 1
        adjacency_matrix = np.zeros((n_labels + 1, n_labels + 1), dtype=np.int32)
        boundary_coords = np.argwhere(distance <= self.sigma + 1)
        con = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for y, x in boundary_coords:
            current_label = labels_map[y, x]
            if current_label == 0:
                continue
            for dy, dx in con:
                ny, nx = y + dy, x + dx
                if ny < 0 or nx < 0 or ny >= labels_map.shape[0] or nx >= labels_map.shape[1]:
                    continue
                neighbor_label = labels_map[ny, nx]
                if neighbor_label != current_label and neighbor_label != 0:
                    adjacency_matrix[current_label - 1, neighbor_label - 1] = 1
                    adjacency_matrix[neighbor_label - 1, current_label - 1] = 1
        self.adjacency_matrix = adjacency_matrix

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
        fig, ax = plt.subplots(figsize=(8, 8))
        for polygon in self.polygons:
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc='r', ec='black')
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
