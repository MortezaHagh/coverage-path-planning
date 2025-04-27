""" Trapezoidal decomposition """
from shapely import Polygon
import matplotlib.pyplot as plt
from scripts.decomposition.events import Event
from scripts.decomposition.obstacles import Obstacle
from scripts.decomposition.decomposition_1_mh_f import MHDecomposition


class BoustraphedonDecomposition(MHDecomposition):
    def __init__(self):
        super().__init__()
        self.ns = "MHDecomposition"
