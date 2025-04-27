from shapely import Polygon
from scripts.decomposition.decomposition_1_mh_f import MHDecomposition
# from scripts.decomposition.decomposition_2_bs_mh_f import BoustraphedonDecomposition


# example
def create_bounding():
    # create boarder polygon
    b_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    return b_poly


def create_polygons():
    # create obstacles polygons
    p1 = Polygon([(40, 60), (70, 60), (80, 66), (70, 70), (50, 80), (40, 70)])
    p2 = Polygon([(30, 30), (40, 20), (60, 30), (60, 50), (30, 50), (20, 40)])
    # p2 = Polygon([(30, 30), (40, 20), (60, 30), (70, 40), (60, 50), (30, 50), (20, 40)])
    return [p1, p2]


def main():
    td = MHDecomposition()
    td.set_bounding_polygon(create_bounding())
    td.set_obstacle_polygons(create_polygons())
    td.initiate()


if __name__ == "__main__":
    main()
