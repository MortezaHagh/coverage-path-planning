import cv2
import numpy as np

# --- Load image (binary map) ---
# map_fr52
dir = '/workspaces/00-CPP/Data/Map4/AMain/merge.pgm'
img = cv2.imread(dir, cv2.IMREAD_GRAYSCALE)

# Make sure it's binary (0 = wall, 255 = free)
_, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# --- Edge detection ---
edges = cv2.Canny(bw, 50, 150, apertureSize=3)

# --- Hough Transform ---
lines = cv2.HoughLines(edges, 1, np.pi/180, 120)  # tweak threshold if needed

angles = []
if lines is not None:
    for rho, theta in lines[:, 0]:
        angles.append(theta)

# Compute dominant wall orientation
dominant_angle = np.median(angles)   # robust estimate
major_direction = (dominant_angle + np.pi/2) % np.pi

print("Dominant wall angle (deg):", np.degrees(dominant_angle))
print("Main free-space direction (deg):", np.degrees(major_direction))

# --- Visualization ---
# Convert grayscale map to BGR for drawing
vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

# Draw arrow from center of map in free-space direction
h, w = bw.shape
center = (w//2, h//2)
length = min(w, h) // 3  # arrow length

dx = int(length * np.cos(major_direction))
dy = int(length * np.sin(major_direction))

cv2.arrowedLine(vis, center, (center[0] + dx, center[1] + dy), (0, 0, 255), 3, tipLength=0.2)

# Show results
cv2.imshow("Map with Direction", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
