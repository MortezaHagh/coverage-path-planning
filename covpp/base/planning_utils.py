""" Utility functions for planning. """
import numpy as np


def rearrange_points_by_point(point: np.ndarray, points: np.ndarray) -> np.ndarray:
    """ Rearrange points starting from the closest point to the given point. """
    # Find the index of the closest point to the given point
    idx = find_closest_point_idx(point, points)
    # Rearrange points starting from the closest point
    rearranged_points = rearrange_points_by_idx(idx, points)
    # add first point to the end of the list
    rearranged_points = np.concatenate((rearranged_points, [rearranged_points[0]]), axis=0)
    return rearranged_points


def rearrange_points_by_idx(idx: int, points: np.ndarray) -> np.ndarray:
    """ Rearrange points starting from idx. """
    if idx < 0 or idx >= len(points):
        print(f"Index {idx} is out of bounds for points list.")
        return points
    if idx == 0:
        return points
    return np.concatenate((points[idx:], points[:idx]), axis=0)


def find_closest_point_idx(p: np.ndarray, points: np.ndarray) -> int:
    """ Find the index of the closest point to p in points. """
    dists = [np.linalg.norm(np.array(p) - np.array(pt)) for pt in points]
    return np.argmin(dists)
