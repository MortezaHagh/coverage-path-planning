""" Hough Transform based main direction finder. """
import cv2
import numpy as np


class HoughDirectionFinder:
    """ Find main direction of a cell using Hough Transform. """

    def __init__(self, map_array: np.ndarray):
        self.map = map_array

    def cell_main_direction(self) -> float:
        """ Find main direction of the cell using Hough Transform. """
        # Edge detection
        edges = cv2.Canny(self.map, 50, 150, apertureSize=3)

        # Hough Transform # tweak threshold if needed
        threshold = 120   # start threshold
        min_threshold = 50  # don't go below this

        lines_p = None
        while threshold >= min_threshold and lines_p is None:
            lines_p = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=80,      # tune for your maps
                minLineLength=20,  # minimum segment length
                maxLineGap=10      # allowable gap
            )
            threshold -= 20  # decrease threshold step-by-step

        if lines_p is not None and len(lines_p) > 0:
            # Find longest segment
            lengths = []
            for x1, y1, x2, y2 in lines_p[:, 0]:
                dx, dy = x2 - x1, y2 - y1
                lengths.append((dx*dx + dy*dy, (x1, y1, x2, y2)))
            _, (x1, y1, x2, y2) = max(lengths, key=lambda t: t[0])

            # Angle of longest line segment
            angle = np.arctan2(y2 - y1, x2 - x1)  # [-π, π]
            major_direction = (angle + np.pi/2) % np.pi
        else:
            major_direction = 0.0

        # # visualize
        self.visualize_main_direction(major_direction, lines_p=lines_p)
        return major_direction

    def cell_main_direction_1(self) -> float:
        """ Find main direction of the cell using Hough Transform. """

        # Edge detection
        edges = cv2.Canny(self.map, 50, 150, apertureSize=3)

        # Hough Transform # tweak threshold if needed
        threshold = 80   # start threshold
        min_threshold = 50  # don't go below this
        lines = None

        while threshold >= min_threshold and lines is None:
            lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi/180, threshold=threshold)
            if lines is None:
                threshold -= 20  # decrease threshold step-by-step

        # If no lines detected, fallback to default angle (e.g., 0)
        angles = []
        if lines is not None and len(lines) > 0:
            # rho, theta
            for _, theta in lines[:, 0]:
                angles.append(theta)
        else:
            # Fallback: assume horizontal direction
            angles.append(0.0)

        # select dominant wall orientation (not median)
        dominant_angle = max(set(angles), key=angles.count)
        major_direction = (dominant_angle) % np.pi

        # # visualize
        self.visualize_main_direction(major_direction, lines=lines)
        return major_direction

    def visualize_main_direction(self, major_direction: float, lines: np.ndarray = None, lines_p: np.ndarray = None) -> float:
        """ Visualize main direction on the cell map. """
        # visualization
        # Draw arrow from center of map in free-space direction
        h, w = self.map.shape
        center = (w//2, h//2)
        length = min(w, h) // 3  # arrow length

        dx = int(length * np.cos(major_direction))
        dy = int(length * np.sin(major_direction))

        # Convert grayscale map to BGR for drawing
        vis = cv2.cvtColor(self.map, cv2.COLOR_GRAY2BGR)

        # add lines to vis by green color
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(vis, pt1, pt2, (0, 255, 0), 2)

        if lines_p is not None:
            for x1, y1, x2, y2 in lines_p[:, 0]:
                cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # draw major direction arrow
        cv2.arrowedLine(vis, center, (center[0] + dx, center[1] + dy), (0, 0, 255), 3, tipLength=0.2)
        # display image, lines, and major direction
        cv2.imshow("Cell Main Direction", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # map_fr52
    dir = '/home/morteza/Documents/00-CovPP/covpp/results/results_r_0.3/map_fr52/District/maps/local_map_3_cleaning.png'
    map_image = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

    hough_finder = HoughDirectionFinder(map_image)
    main_direction = hough_finder.cell_main_direction()
    # main_direction = hough_finder.cell_main_direction_1()
    print(f"Main Direction (radians): {main_direction}")

    # # Visualize
    # hough_finder.vizualize_main_direction(main_direction)
