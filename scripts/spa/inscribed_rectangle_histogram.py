import cv2 as cv
import numpy as np
FREE_COST = 255
OCCUPIED_COST = 0


def largest_rectangle_in_histogram(heights):
    # Find the largest rectangle in a histogram
    stack = []
    max_area = 0
    best_rect = None

    for i, h in enumerate(heights + [0]):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            area = height * width
            if area > max_area:
                max_area = area
                best_rect = (stack[-1] + 1 if stack else 0, i - 1)
        stack.append(i)

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
        row, (start_col, end_col) = best_rect
        top_left = (start_col, row - height[row, start_col] + 1)
        bottom_right = (end_col, row)
        #
        xc = (top_left[0] + bottom_right[0]) / 2
        yc = (top_left[1] + bottom_right[1]) / 2
        w = bottom_right[0] - top_left[0] + 1
        h = bottom_right[1] - top_left[1] + 1
        # rot_deg = 0
        # data = {"xc": xc, "yc": yc, "rot": np.deg2rad(rot_deg), "w": w, "h": h}

        return top_left, bottom_right

    return None


def find_largest_in_rectangle(map_i: np.ndarray) -> dict:
    """ find largest rectangle in the biggest open space contour """
    # Find contours
    contours, _ = cv.findContours(map_i, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv.contourArea)

    # Create a mask from the largest contour
    mask = np.zeros_like(map_i)
    cv.drawContours(mask, [largest_contour], -1, FREE_COST, thickness=cv.FILLED)

    # Find the largest rectangle inside the mask
    # Use the mask to find the largest inscribed rectangle
    largest_rectangle = find_largest_inscribed_rectangle(mask)

    return largest_rectangle
