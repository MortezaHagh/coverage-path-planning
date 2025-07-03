import numpy as np
import shapely
from shapely.geometry import MultiPolygon, MultiPoint, Polygon
from matplotlib import pyplot as plt
# create a polygon


def create_polygon(coords):
    return Polygon(coords)

# plot polygon, its centroid, and max inscribed circle


def plot_polygon(poly):
    fig, ax = plt.subplots()
    x, y = poly.exterior.xy
    ax.plot(x, y, color='blue', label='Polygon')

    # plot centroid
    centroid = poly.centroid
    ax.plot(centroid.x, centroid.y, 'ro', label='Centroid')

    # ploy mic
    # Returns a two-point linestring, with the first point at the center of the inscribed circle and the second on the boundary of the inscribed circle
    mic = shapely.maximum_inscribed_circle(poly)
    mic_center, mic_boundary = mic.coords
    ax.plot(mic_center[0], mic_center[1], 'go', label='MIC Center')
    radius = np.linalg.norm(np.array(mic_center) - np.array(mic_boundary))
    circle = plt.Circle(mic_center, radius, color='green', fill=False, label='MIC Radius')
    ax.add_artist(circle)

    # minimum_bounding_circle
    # Returns polgon
    mbc_polygon = shapely.minimum_bounding_circle(poly)
    mbc_x, mbc_y = mbc_polygon.exterior.xy
    ax.plot(mbc_x, mbc_y, color='orange', label='Minimum Bounding Circle')

    # minimum_rotated_rectangle polygon
    mrr = poly.minimum_rotated_rectangle
    mrr_x, mrr_y = mrr.exterior.xy
    ax.plot(mrr_x, mrr_y, color='purple', label='Minimum Rotated Rectangle')

    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    coords = [(0, 0), (4, 0),  (6, -5), (6, 5),   (4, 3), (2, 5), (0, 3)]
    poly = create_polygon(coords)

    print("Polygon Area:", poly.area)
    print("Polygon Centroid:", poly.centroid)

    plot_polygon(poly)

    # Check if the polygon is valid
    print("Is the polygon valid?", poly.is_valid)
