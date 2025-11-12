""" Cell class for voronoi-based decomposition. """
import numpy as np
from shapely import Polygon
from covpp.base.logger import MyLogger
from covpp.base.configs import PlanningConfig, PipeLineConfig
from covpp.segmentation.cell_base import CellBase
from covpp.segmentation.voronoi_segmentation.common import Ring  # , CriticalPoint


class Cell (CellBase):
    """ Cell for voronoi-based decomposition.

    Processes:
        Find the minimum clearance of the cell based on critical points.
        Process the cell:
        - Removing points that are too close to critical points.

    Inputs:
        Rings (closed loops) containing critical points.
    """

    def __init__(self, ind: int, ring: Ring, resolution: float = None) -> None:
        super().__init__(ind, ring.poly, resolution)
        self.lg = MyLogger(f"Cell-{self.id}")

        # attributes
        self.ring: Ring = ring                  # rings (closed loops) containing critical points
        self.basic: bool = False                # if no critical points, then basic cell
        self.min_clearance: float = 0.0         # minimum clearance of the cell based on critical points

        # settings
        if resolution is None:
            raise ValueError("Resolution must be provided for the cell.")
        self.robot_r_px = int(PlanningConfig.cleaning_r_ov / resolution)

        # find the minimum clearance of the cell
        self.find_min_clearance()

    def find_min_clearance(self) -> None:
        """ Find the minimum clearance of the cell based on critical points. """
        if not self.ring.cp_points:
            self.basic = True
            self.lg.warn("No critical points set for the cell, marking as basic.")
            return
        # cp: CriticalPoint
        clearances = [cp.vp.clearance for cp in self.ring.cp_points]
        self.min_clearance = min(clearances)

    def process_cell(self) -> None:
        """ Main function to process the cell.
        1- Modify the cell by removing points that are too close to critical points.
        """
        if PipeLineConfig().method in ["contour"]:
            self.modify_cell()

    def modify_cell(self) -> None:
        """ Modify the cell by removing points that are too close to critical points. """

        # minimum distance to critical points
        min_d = self.robot_r_px * 1.0

        # method 2
        n = self.ring.n_p
        to_rm = []
        for i in self.ring.cp_ring_idxs:            # index of critical point in the ring
            for c, ii in zip([-1, 1], [i, i+1]):    # check previous and next points
                cp = self.ring.points[ii]
                js = [(ii+c) % n, (ii+(2*c)) % n]                   # indices of the next (or previous) two points
                cjs = np.array([self.ring.points[j] for j in js])   # coordinates of the next two points
                ds = np.linalg.norm(cp - cjs, axis=1)               # distances to the next two points
                if ds[0] < min_d:
                    to_rm.append(ii)
                    if ds[1] < min_d:
                        to_rm.append(js[0])
        #
        new_points = [self.ring.points[i] for i in range(n) if i not in to_rm]

        # # method 1
        # cell_points = self.poly.exterior.coords
        # new_points = []
        # for p in cell_points:
        #     p_valid = True
        #     for cp in self.ring.cp_points:
        #         if cp.n_passed > 1:
        #             dcp = np.linalg.norm(np.array(p) - np.array(cp.vp.v))
        #             d1 = np.linalg.norm(np.array(p) - np.array(cp.bs1_vp))
        #             d2 = np.linalg.norm(np.array(p) - np.array(cp.bs2_vp))
        #             d = min(dcp, d1, d2)
        #             if d < min_d:
        #                 p_valid = False
        #                 break
        #     if p_valid:
        #         new_points.append(p)
        #
        if len(new_points) > 3:
            self.poly = Polygon(new_points)
            self.create_cell_points()
