""" Trapezoidal decomposition """
from shapely import Polygon
import matplotlib.pyplot as plt
from scripts.decomposition.events import Event
from scripts.decomposition.obstacles import Obstacle
from scripts.decomposition.decomposition_1_tp_mh_f import TrapezoidalDecomposition


class BoustraphedonDecomposition(TrapezoidalDecomposition):
    def __init__(self):
        super().__init__()
        self.ns = "TrapezoidalDecomposition"
