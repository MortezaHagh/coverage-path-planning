""" Find the largest inscribed rectangle in a binary map using histogram method. """
import cv2 as cv
import numpy as np
FREE_COST = 255
OCCUPIED_COST = 0


def largest_rectangle_in_histogram(heights):
    # Find the largest rectangle in a histogram
    stack = []
    max_area = 0
    best_height = None
    best_rect = None

    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            w_start = stack[-1] if stack else 0
            w_end = i-1
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                best_height = height
                best_rect = (w_start, w_end, best_height)
                # best_rect = (stack[-1] + 1 if stack else 0, i - 1)
        stack.append(i)

    # while stack:
    #     height = heights[stack.pop()]
    #     w_start = stack[-1] if stack else 0
    #     w_end = len(heights)-1
    #     width = len(heights)-1 if not stack else len(heights) - stack[-1] - 1
    #     area = height * width
    #     if area > max_area:
    #         max_area = area
    #         best_height = height
    #         best_rect = (w_start, w_end, best_height)

    return max_area, best_rect


def find_largest_inscribed_rectangle(mask):
    # Use dynamic programming to find the largest rectangle
    # Create an array to store the maximum height of 1s at each pixel
    height = np.zeros_like(mask, dtype=np.int32)
    rows, cols = mask.shape

    for i in range(rows):
        for j in range(cols):
            if mask[i, j] == FREE_COST:  # Open space
                height[i, j] = height[i - 1, j] + 1 if i > 0 else 1

    # Find the largest rectangle using a histogram-based approach
    max_area = 0
    best_rect = None

    for i in range(rows):
        max_area_row, rect = largest_rectangle_in_histogram(height[i])
        if max_area_row > max_area:
            max_area = max_area_row
            best_rect = (i, rect)

    # Convert the best rectangle to coordinates
    if best_rect:
        row, (start_col, end_col, best_height) = best_rect
        top_left = (start_col, row - best_height)
        bottom_right = (end_col, row)
        # top_left = (start_col, row - height[row, start_col] + 1)

        # # Validate that the rectangle is fully within the free space
        # for y in range(top_left[1], bottom_right[1] + 1):
        #     for x in range(top_left[0], bottom_right[0] + 1):
        #         if mask[y, x] != FREE_COST:  # If any part is not free, discard
        #             return None

        xc = int((top_left[0] + bottom_right[0]) / 2)
        yc = int((top_left[1] + bottom_right[1]) / 2)
        w = int(bottom_right[0] - top_left[0] + 1)
        h = int(bottom_right[1] - top_left[1] + 1)
        rot_deg = 0
        data = {"xc": xc, "yc": yc, "rot": np.deg2rad(rot_deg), "w": w, "h": h}

        return data

    return None


def find_largest_in_rectangle(map_i: np.ndarray) -> dict | None:
    """ find largest rectangle in the biggest open space contour. """
    # Find contours
    contours, _ = cv.findContours(map_i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros_like(map_i)
    cv.drawContours(mask, [largest_contour], -1, FREE_COST, thickness=cv.FILLED)

    # # display contours
    # color_image = cv.cvtColor(map_i, cv.COLOR_GRAY2BGR)
    # cv.drawContours(color_image, contours, -1, (0, 0, 255), thickness=5)
    # cv.imshow("Contours in Red", color_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # cv.imwrite("Contours.png", color_image)

    # Find the largest rectangle inside the mask
    # Use the mask to find the largest inscribed rectangle
    largest_rectangle_data = find_largest_inscribed_rectangle(mask)
    data = largest_rectangle_data
    # data = {"xc": xc, "yc": yc, "rot": np.deg2rad(rot_deg), "w": w, "h": h}

    if data is None:
        print("[find_largest_inscribed_rectangle]: Failed to find a valid rectangle")
        return None
        # raise ValueError("Failed to find a valid rectangle")

    #  largest_rectangle
    rotated_rect = ((data["xc"], data["yc"]), (data["w"], data["h"]), data["rot"])
    vertices = cv.boxPoints(rotated_rect)
    vertices = np.int32(vertices)
    vertices = vertices.tolist()

    return {"vertices": vertices, "rect_data": data}


def find_inscribed_rotated_rectangle(map_i: np.ndarray) -> dict | None:
    return find_largest_in_rectangle(map_i)
