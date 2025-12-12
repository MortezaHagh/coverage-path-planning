""" Data structures for skeleton segmentation."""
import dataclasses
import numpy as np


@dataclasses.dataclass
class Skelet:
    """ Contains skeletonization results. """
    skeleton_map: np.ndarray                # dtype=bool
    distance_map: np.ndarray                # dtype=uint8
    distance_norm_map: np.ndarray           # dtype=uint8 OCCUPIED-FREE
    distance_skeleton_map: np.ndarray       # dtype=uint8 OCCUPIED-FREE
    skeleton_segmented_map: np.ndarray      # dtype=uint8
    end_points: np.ndarray                  # dtype=uint8 (y, x)
    branch_points: np.ndarray               # dtype=uint8 (y, x)
    end_points_filtered: np.ndarray         # dtype=uint8 (y, x)


@dataclasses.dataclass
class Branch:
    """ Represents a skeleton branch/path between end_points/branch_points. """
    length: int                     # length of the branch (number of points)
    end_type: str                   # 'endpoint' | 'branchpoint' | 'skip'
    end_point: tuple                # (y, x)
    start_point: tuple              # (y, x)
    points: np.ndarray              # shape (N, 2), dtype=int , list of (y, x)
    cp_idxs: np.ndarray = None      # indices in critical points list
    cp_coords: np.ndarray = None    # shape (N, 2), dtype=int , list of (y, x)
