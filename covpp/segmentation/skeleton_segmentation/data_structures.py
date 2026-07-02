""" Data structures for skeleton segmentation."""
import dataclasses
import numpy as np
from shapely.geometry import Polygon


@dataclasses.dataclass
class Configs:
    """Configuration parameters for skeleton decomposition."""
    end_dist_threshold_ratio: float = 0.0  # minimum ratio of distance from branch ends to overall min-max distance in branch to consider critical point
    max_critical_region_ratio: float = 0.6  # maximum ratio of critical region length to branch length
    min_max_distance_ratio: float = 0.6     # minimum ratio of min distance to max distance in branch to consider critical point
    min_area_thresh_for_contour: int = 10   # minimum area threshold for valid contour


@dataclasses.dataclass
class Skelet:
    """ Contains skeletonization results. """
    skeleton_map_b: np.ndarray              # dtype=bool
    distance_map: np.ndarray           # dtype=uint8 OCCUPIED-FREE
    skeleton_distance_map: np.ndarray       # dtype=uint8 OCCUPIED-FREE
    skeleton_segmented_map: np.ndarray      # dtype=uint8
    end_points: np.ndarray                  # dtype=uint8 (y, x)
    branch_pts: np.ndarray               # dtype=uint8 (y, x)
    f_end_pts: np.ndarray         # dtype=uint8 (y, x)


@dataclasses.dataclass
class Branch:
    """ Represents a skeleton branch/path between end_points/branch_pts. """
    start_point: tuple              # (y, x)
    end_type: str                   # 'endpoint' | 'branchpoint' | 'skip'
    points: np.ndarray              # shape (N, 2), dtype=int , list of (y, x)
    length: int                     # length of the branch (number of points)
    end_point: tuple                # (y, x)
    # cp_idxs: np.ndarray = None      # indices in critical points list
    # cp_coords: np.ndarray = None    # shape (N, 2), dtype=int , list of (y, x)


class CriticalPoint:
    """Critical point data structure."""

    def __init__(self, coord: np.ndarray):
        self.coord = coord  # (y, x)
        self.cp_skel_segment: int = -1  # skeleton segment of critical point
        self.side_sk_segments: tuple[int] = ()  # skeleton segments of side points
        self.side_points: list[tuple[int, int]] = []  # side points (y, x)
        self.base_pts: list[tuple[int, int]] = []  # base points (y, x)
        self.base_paths: list[tuple[tuple[int, int]]] = []  # paths from base points to critical point


class SegmentSolution:
    """Solution for segmentation."""

    def __init__(self, segmented_map: np.ndarray, seg_ids: list[int], polygons: list[Polygon]):
        self.segmented_map = segmented_map
        self.seg_ids = seg_ids
        self.polygons = polygons
