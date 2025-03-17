def largest_rectangle_in_histogram(heights):
    stack = []
    max_area = 0
    heights.append(0)  # Add a sentinel value to flush the stack at the end
    best_rect = (0, 0)
    for i, h in enumerate(heights):
        print("h", h)
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
            if max_area == height * width:
                best_rect = (stack[-1] + 1 if stack else 0, i - 1)

        stack.append(i)
    heights.pop()  # Remove the sentinel value
    return max_area, best_rect


def largest_rectangle_in_binary_matrix(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    histogram = [0] * cols
    max_area = 0
    g_max_area = 0

    for row in matrix:
        # Update the histogram for the current row
        for col in range(cols):
            histogram[col] = histogram[col] + 1 if row[col] == 1 else 0

        print("=========")
        print(histogram)
        # Find the largest rectangle in the current histogram
        max_area, best_rect = largest_rectangle_in_histogram(histogram)
        print("max_area", max_area)
        g_max_area = max(max_area, g_max_area)

    print("g_max_area", g_max_area)
    print("best_rect", best_rect)

    return max_area


# Example usage
binary_matrix = [
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0]
]

max_rectangle_area = largest_rectangle_in_binary_matrix(binary_matrix)
print(f"Largest rectangle area: {max_rectangle_area}")
