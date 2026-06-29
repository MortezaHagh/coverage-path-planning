""" Data structures for skeleton segmentation."""
import dataclasses
import numpy as np
from shapely.geometry import Polygon  # , LineString


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


def find_level(h, hierarchy):
    """ Find the level of contour in hierarchy. """
    level = 0
    parent = h[3]
    while parent != -1:
        level += 1
        parent = hierarchy[parent][3]
    return level


def is_valid_contour(h, hierarchy) -> bool:
    # if level is odd, it is a hole contour
    level = find_level(h, hierarchy)
    if level % 2 == 1:
        return False
    return True
    # # Check if contour has both parent and children in hierarchy.
    # return not(h[3] != -1 and h[2] != -1)
