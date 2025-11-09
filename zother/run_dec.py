from shapely import Polygon
from covpp.planning.boustrophedon_base.decomposition_bs_mh import MHDecomposition
# from covpp.decomposition.decomposition_2_bs_mh_f import BoustraphedonDecomposition

# example


class MyEnv0Simple:
    def __init__(self):
        self.create_bounding()
        self.create_polygons()

    def create_bounding(self):
        # create boarder polygon
        # b_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
        b_poly = Polygon([(0, 50), (10, 0), (100, 0), (110, 50), (100, 100), (10, 100)])
        return b_poly

    def create_polygons(self):
        # create obstacles polygons
        p1 = Polygon([(30, 30), (40, 20), (60, 30), (70, 40), (60, 50), (30, 50), (20, 40)])
        return [p1]


class MyEnv1:
    def __init__(self):
        self.create_bounding()
        self.create_polygons()

    def create_bounding(self):
        # create boarder polygon
        b_poly = Polygon([(0, 90), (10, 80), (0, 70), (0, 50), (-10, 40), (0, 30),
                          (0, 0), (100, 0), (100, 25), (90, 35),
                          (100, 45), (100, 55), (100, 65), (110, 75), (100, 85), (100, 100), (0, 100)])
        return b_poly

    def create_polygons(self):
        # create obstacles polygons
        p1 = Polygon([(40, 60), (70, 60), (80, 66), (70, 70), (50, 90), (40, 80), (40, 70), (50, 65)])
        p2 = Polygon([(30, 30), (40, 20), (60, 25), (50, 35), (60, 45), (60, 50), (30, 50), (20, 40)])
        return [p1, p2]


def main():
    # model = MyEnv0Simple()
    model = MyEnv1()
    td = MHDecomposition()
    td.set_bounding_polygon(model.create_bounding())
    td.set_obstacle_polygons(model.create_polygons())
    td.initiate()


if __name__ == "__main__":
    main()
