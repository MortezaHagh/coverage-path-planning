""" Utility functions for SPA (Spatial Analysis) involving rectangle detection in maps."""
import cv2 as cv
import numpy as np


def find_circumscribed_rectangle(map_i: np.ndarray) -> dict:
    """
    Find the minimum area rectangle that circumscribes all free cells in the map.

    Method:
        1. Find contours of the free cells in the map.
        2. Combine all contours to get the minimum area rectangle using cv.minAreaRect.
    Returns:
        dict: {"vertices": [(x, y), ...], "rect_data": {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}}

    """

    # Find contours of the free cells
    contours, _ = cv.findContours(map_i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Combine all contours to get the minimum area rectangle
    all_points = np.vstack(contours)
    rect: tuple = cv.minAreaRect(all_points)  # rect = (center, (width, height), angle)
    xc = int(rect[0][0])
    yc = int(rect[0][1])
    w = int(rect[1][0])
    h = int(rect[1][1])
    rot_deg = rect[2]
    if h > w:
        w, h = h, w
        rot_deg += 90

    data = {"xc": xc, "yc": yc, "rot": np.deg2rad(rot_deg), "w": w, "h": h}
    vertices: np.ndarray = cv.boxPoints(rect)
    vertices = np.int32(vertices)
    # vertices: [(x, y), ...]
    # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
    return {"vertices": vertices, "rect_data": data}


def find_inscribed_rotated_rectangle_1(map_i: np.ndarray) -> dict:
    """
    Find the maximum area rectangle that can be inscribed within the free cells in the map.

    Method:
        1. Find contours of the free cells in the map.
        2. Find the largest contour and fit a rotated rectangle around it.
        3. Extract the center, size, and angle of the minimum area rectangle.
        4. For various scales and rotations of the rectangle:
        a. Scale the rectangle.
        b. Rotate the rectangle.
        c. Check if all vertices of the rectangle lie within the largest contour.
        d. If yes, calculate the area and update the best rectangle if it's larger than the previous best.
        5. Return the best rectangle found.

    Returns:
        dict: {"vertices": [(x, y), ...], "rect_data": {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}}
    """

    # Find contours
    contours, _ = cv.findContours(map_i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)

    # Find centroid of the contour
    mom = cv.moments(largest_contour)
    mcx = int(mom["m10"] / mom["m00"])
    mcy = int(mom["m01"] / mom["m00"])

    # Compute image moments for the white cells (255 values)
    moments = cv.moments(map_i)
    if moments["m00"] != 0:  # Avoid division by zero
        cx_img = int(moments["m10"] / moments["m00"])  # X-coordinate
        cy_img = int(moments["m01"] / moments["m00"])  # Y-coordinate
        mcx = cx_img
        mcy = cy_img

    # Fit a minimum area rectangle around the contour
    # rect = (center, (width, height), angle)
    rect: tuple = cv.minAreaRect(largest_contour)
    w = int(rect[1][0])
    h = int(rect[1][1])
    rot_deg = rect[2]
    if h > w:
        w, h = h, w
        rot_deg += 90
    min_area_rect = ((mcx, mcy), (w, h), rot_deg)

    # Initialize variables
    max_area = 0
    best_rectangle = None
    best_vertices = None

    # Extract the center, size, and angle of the minimum area rectangle
    center, size, angle = min_area_rect

    # Move step for adjusting the rectangle position
    move_step = 2

    # Iterate over possible sizes and angles
    for c in [-1, 1]:
        success = False
        for rotation in np.linspace(angle, angle + 45, 90):
            for scale_w in np.linspace(1.0, 0.1, 50):
                for scale_h in np.linspace(1.0, 0.1, 50):

                    # Scale the rectangle
                    scaled_size = (size[0] * scale_w, size[1] * scale_h)

                    # Create a rotated rectangle
                    rotated_rect = ((center[0], center[1]), scaled_size, rotation*c)

                    # Check if the rectangle is inside the contour
                    vertices = cv.boxPoints(rotated_rect)
                    vertices = np.int32(vertices)
                    vertices = vertices.tolist()

                    if all(cv.pointPolygonTest(largest_contour, tuple(pt), False) > 0 for pt in vertices):
                        # Calculate the area of the rectangle
                        area = scaled_size[0] * scaled_size[1]
                        success = True
                        if area > max_area:
                            max_area = area
                            best_rectangle = rotated_rect
                            best_vertices = vertices
                    else:
                        # Move the center in the opposite direction of the point of contact
                        for pt in vertices:
                            if cv.pointPolygonTest(largest_contour, tuple(pt), False) <= 0:
                                direction = np.array(center) - np.array(pt)
                                direction = direction / np.linalg.norm(direction) * move_step
                                new_center = np.array(center) + direction
                                new_center = new_center.astype(int).tolist()
                                # Create a rotated rectangle
                                new_rect = ((new_center[0], new_center[1]), scaled_size, rotation)
                                vertices = cv.boxPoints(new_rect)
                                vertices = np.int32(vertices)
                                vertices = vertices.tolist()
                                if all(cv.pointPolygonTest(largest_contour, tuple(pt), False) > 0 for pt in vertices):
                                    center = new_center
                                break
            if success:
                break

    if best_rectangle:
        xc = int(best_rectangle[0][0])
        yc = int(best_rectangle[0][1])
        w = int(best_rectangle[1][0])
        h = int(best_rectangle[1][1])
        rotation = best_rectangle[2]
        data = {"xc": xc, "yc": yc, "rot": np.deg2rad(rotation), "w": w, "h": h}
        vertices = np.int32(best_vertices)
        # vertices: [(x, y), ...]
        # data = {"xc": xc, "yc": yc, "rot": rot, "w": w, "h": h}
        return {"vertices": vertices, "rect_data": data}

    raise ValueError("No rectangle found")


def find_inscribed_rotated_rectangle(map_i: np.ndarray) -> dict:
    return find_inscribed_rotated_rectangle_1(map_i)
