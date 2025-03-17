import cv2 as cv
import numpy as np


def find_circumscribed_rectangle(map_i: np.ndarray) -> dict:
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


def find_inscribed_rotated_rectangle(map_i: np.ndarray):  # -> dict[str, float]:

    # Find contours
    contours, _ = cv.findContours(map_i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)

    # find centroid of the contour
    mom = cv.moments(largest_contour)
    mcx = int(mom["m10"] / mom["m00"])
    mcy = int(mom["m01"] / mom["m00"])

    # # Display the image with contours
    # m = map_i.copy()
    # cv.drawContours(m, [largest_contour], -1, 100, 10)
    # cv.imshow("Contours", m)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # Fit a minimum area rectangle around the contour
    # rect = (center, (width, height), angle)
    rect: tuple = cv.minAreaRect(largest_contour)
    xc = int(rect[0][0])
    yc = int(rect[0][1])
    w = int(rect[1][0])
    h = int(rect[1][1])
    rot_deg = rect[2]
    if h > w:
        w, h = h, w
        rot_deg += 90
    # min_area_rect = ((xc, yc), (w, h), rot_deg)
    min_area_rect = ((mcx, mcy), (w, h), rot_deg)

    # Initialize variables
    max_area = 0
    rotation = 0
    data = None
    vertices = None
    best_rectangle = None
    best_vertices = None
    sol = None

    # Extract the center, size, and angle of the minimum area rectangle
    center, size, angle = min_area_rect

    # Iterate over possible sizes and angles
    for scale_w in np.linspace(1.0, 0.1, 50):  # Scale the rectangle size
        for scale_h in np.linspace(1.0, 0.1, 50):  # Scale the rectangle size
            success = False
            for rotation in np.linspace(angle, angle + 90, 50):  # Rotate around the contour's angle
                # Scale the rectangle
                scaled_size = (size[0] * scale_w, size[1] * scale_h)

                # Create a rotated rectangle
                rotated_rect = ((center[0], center[1]), scaled_size, rotation)

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
        sol = {"vertices": vertices, "rect_data": data}

    return sol
