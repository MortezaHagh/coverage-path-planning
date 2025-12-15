""" Utility classes and functions for geometric primitives."""
import dataclasses
import numpy as np
from shapely import Polygon


@dataclasses.dataclass
class MyPoint:
    """Simple 2D point representation.

    Attributes:
        x: X-coordinate (float or int).
        y: Y-coordinate (float or int).
        xy: Property returning (x, y) tuple.
    """

    x: float | int = None
    y: float | int = None

    @property
    def xy(self):
        """Return point as (x, y) tuple."""
        return (self.x, self.y)


class MyPath:
    """Path representation as sequence of MyPoint objects.

    This class provides a container for storing and manipulating
    sequences of 2D points representing paths.

    Attributes:
        points: List of MyPoint objects.
        xy: NumPy array of (x, y) coordinates.
        n: Number of points in the path.
    """

    def __init__(self, points: list[MyPoint] = None):
        self.points: list[MyPoint] = points
        self.xy: np.ndarray = []
        self.n: int = 0
        self.set_xys()

    def set_points(self, points: list[MyPoint]) -> None:
        """Set new list of points and update xy array.

        Args:
            points: List of MyPoint objects.
        """
        self.points = points
        self.set_xys()

    def set_xys(self):
        """Create xy numpy array from points list.

        Automatically detects appropriate dtype (float32 or int32) based on
        the coordinate types.
        """
        if self.points is not None:
            xy = [(p.x, p.y) for p in self.points]
            if isinstance(xy[0][0], (float, np.float32)):
                self.xy = np.array(xy, dtype=np.float32)
            else:
                self.xy = np.array(xy, dtype=np.int32)
            self.n = len(self.points)

    def point(self, i: int) -> MyPoint:
        """Get point at specified index.

        Args:
            i (int): Index of point to retrieve.

        Returns:
            MyPoint at index i.

        Raises:
            IndexError: If index is out of range.
        """
        if i < 0 or i >= self.n:
            raise IndexError("Index out of range")
        return self.points[i]


def mod_angle(theta: float) -> float:
    """ Normalize angle to [0, 2π)."""
    return theta % (2 * np.pi)


def angle_diff(a, b):
    """
    Returns the smallest signed difference between two angles (in radians).
    Result is in range [-pi, pi).
    """
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return diff


def find_longest_edge_poly(polygon: Polygon) -> dict:
    """ Find the longest edge of a polygon and return its properties."""
    # find biggest edge of the polygon
    coords = np.array(polygon.exterior.coords)
    edges = np.diff(coords, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    max_edge_idx = np.argmax(edge_lengths)
    edge_length = edge_lengths[max_edge_idx]

    # two edge points
    p1 = coords[max_edge_idx]
    p2 = coords[(max_edge_idx + 1) % len(coords)]

    # calculate the angle of the edge p1->p2
    theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    # return results
    result = {
        'p1': p1,
        'p2': p2,
        'angle': theta,
        'length': edge_length
    }
    return result


def find_shortest_edge_poly(polygon: Polygon) -> dict:
    """ Find the shortest edge of a polygon and return its properties."""
    # find smallest edge of the polygon
    coords = np.array(polygon.exterior.coords)
    edges = np.diff(coords, axis=0)
    edge_lengths = np.linalg.norm(edges, axis=1)
    min_edge_idx = np.argmin(edge_lengths)
    edge_length = edge_lengths[min_edge_idx]

    # two edge points
    p1 = coords[min_edge_idx]
    p2 = coords[(min_edge_idx + 1) % len(coords)]

    # calculate the angle of the edge p1->p2
    theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    # return results
    result = {
        'p1': p1,
        'p2': p2,
        'angle': theta,
        'length': edge_length
    }
    return result


def find_cnt_level(h, hierarchy) -> int:
    """ Find the level of a contour in the hierarchy."""
    level = 0
    while h[3] != -1:
        h = hierarchy[0][h[3]]
        level += 1
    return level


def evaluate_polygon(polygon: Polygon) -> tuple[bool, str]:
    """ Check if a polygon is valid (non-empty, has area, and is valid).

    Returns:
        tuple[bool, str]: (is_valid, status) where status explains invalidity.
    """
    is_valid = True
    status = ""
    if not polygon.is_valid:
        is_valid = False
        status = "invalid"
    elif polygon.is_empty:
        is_valid = False
        status = "empty"
    elif not polygon.is_simple:
        is_valid = False
        status = "not_simple"
    elif polygon.area == 0:
        is_valid = False
        status = "zero_area"

    if not is_valid:
        return False, status

    # check if it is multi polygon
    if polygon.geom_type == 'MultiPolygon':
        is_valid = False
        status = "multi_polygon"

    return is_valid, status
