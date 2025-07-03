import numpy as np
from shapely import Polygon
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi  # pylint: disable=no-name-in-module
from scripts.base.logger import MyLogger
from scripts.base.configs import DistrictConfgs
from scripts.planning.contour.cell import Cell
from scripts.planning.map_polygons import MapPolygons
from scripts.district.zone_map_data import ZoneMapData
from scripts.planning.contour.common import CriticalPoint
from scripts.planning.optimize_mcells_seq_1 import OptimCellsSeq
from scripts.planning.contour.decomposition_voronoi import VoronoiSegmentation


class DecompositionInterface:
    """Interface for voronoi decomposition methods."""

    def __init__(self, zmd: ZoneMapData, map_polygs: MapPolygons):
        self.ns = "DecompositionInterface"
        self.lg = MyLogger(self.ns)
        #
        self.dcfgs: DistrictConfgs = DistrictConfgs()
        self.zmd: ZoneMapData = zmd
        self.map_polygs: MapPolygons = map_polygs
        self.vor: Voronoi = None
        #
        self.mcells: list[Cell] = []
        self.smcells: list[Cell]
        self.seq_idxs: list[int] = []
        self.n_mcells: int = 0

        #
        self.poly_start_idxs: list[int] = []  # start indices of the polygons
        self.poly_end_idxs: list[int] = []  # end indices of the polygons
        self.critical_points: list[CriticalPoint] = []
        self.cp_base1_idxs: list[int] = []
        self.cp_base2_idxs: list[int] = []
        # segments created from critical points
        self.rings: list[np.ndarray] = []

        #
        # self.critp_ring_dict: dict[int, set[int]] = {}  # critical point index to ring indices mapping
        # ring index to critical point indices mapping
        self.ring_critp_dict: dict[int, set[int]] = {}

        # run the decomposition
        self.run()

    def run(self) -> None:
        # voronoi processing
        vor_dec = VoronoiSegmentation(self.zmd, self.map_polygs)
        vor_dec.voronoi_decomposition()
        self.vor = vor_dec.vor
        self.critical_points = vor_dec.critical_points

        # segmentation
        if self.check_critical_points(vor_dec.b_poly):
            self.main_segmentation(vor_dec.b_poly, vor_dec.obs_polys)

        # create cells
        self.create_cells()
        # self.eval_cells()
        self.n_mcells = len(self.mcells)

        # plotting
        self.plot_rings()

    def main_segmentation(self, b_poly: Polygon, obs_polys: list[Polygon]) -> None:
        # post process
        self.create_poly_points_start_ends(b_poly, obs_polys)
        self.create_critical_idxs()
        self.process_loop_basis()
        # rings
        self.create_all_segments()

    def check_critical_points(self, b_poly: Polygon) -> bool:
        if len(self.critical_points) > 0:
            return True
        self.lg.warn("No critical points found in the Voronoi decomposition.")
        self.rings.append(np.array(b_poly.exterior.coords))
        return False

    def create_poly_points_start_ends(self, b_poly: Polygon, obs_polys: list[Polygon]) -> None:
        poly_start_idxs = [0]
        poly_end_idxs = [len(b_poly.exterior.coords)-1 - 1]
        for poly in obs_polys:
            n_p = len(poly.exterior.coords)
            if n_p < 3:
                self.lg.warn("Skipping polygon with less than 3 points.")
                continue
            poly_start_idxs.append(poly_end_idxs[-1] + 1)
            poly_end_idxs.append(poly_end_idxs[-1] + n_p - 1)
        self.poly_start_idxs = poly_start_idxs
        self.poly_end_idxs = poly_end_idxs

    def create_critical_idxs(self) -> None:
        self.cp_base1_idxs = [c.bs1_idx for c in self.critical_points]
        self.cp_base2_idxs = [c.bs2_idx for c in self.critical_points]
        # initialize critical point dictionary
        # self.critp_ring_dict = {i: set() for i in range(len(self.critical_points))}

    def process_loop_basis(self) -> None:
        """ find critical points that if starintg from them, """
        for i in range(1, len(self.poly_start_idxs)):
            n_bs = 0
            start_idx = self.poly_start_idxs[i]
            end_idx = self.poly_end_idxs[i]
            cp_idx = None
            for j in range(start_idx, end_idx + 1):
                if j in self.cp_base1_idxs:
                    cp_idx = self.cp_base1_idxs.index(j)
                    n_bs += 1
                elif j in self.cp_base2_idxs:
                    cp_idx = self.cp_base2_idxs.index(j)
                    n_bs += 1
            #
            if n_bs == 1:
                self.critical_points[cp_idx].skip = True
                self.lg.warn(f"Critical point {cp_idx} is skipped.")

    def create_all_segments(self) -> None:
        n_ring = 0
        for _ in range(2):
            for i, cp in enumerate(self.critical_points):
                if cp.n_passed > 1 or cp.skip:
                    continue

                # create segment for the critical point
                ring, ring_cp_idxs_set = self.create_cp_segment(i, cp)
                if ring is None:
                    self.lg.warn(f"Critical point {i} with vp.idx {cp.vp.idx} segment creation failed, None ring.")
                    continue
                self.rings.append(ring)

                # ring and critical point mappings
                self.ring_critp_dict[n_ring] = ring_cp_idxs_set
                # for cp_idx in ring_cp_idxs_set:
                #     self.critp_ring_dict[cp_idx].add(n_ring)

                n_ring += 1
        self.lg.info(f"Created {len(self.rings)} rings.")

    def create_cp_segment(self, i_cp: int, cp: CriticalPoint) -> np.ndarray:
        n_points = len(self.vor.points)
        if cp.n_passed > 2:
            raise ValueError("Critical point has been passed more than twice, this should not happen.")

        #
        ring = [cp.vp.v]
        ring_cp_idxs = [i_cp]
        ring_cp_ring_idxs = [0]

        #
        if cp.passed_21:
            ring.append(cp.bs2_vp)
            next_idx = cp.bs2_idx
            cp.passed_12 = True
        else:
            ring.append(cp.bs1_vp)
            next_idx = cp.bs1_idx
            cp.passed_21 = True
        cp.n_passed += 1

        # next_idx
        if next_idx in self.poly_end_idxs:
            end_idx = self.poly_end_idxs.index(next_idx)
            next_idx = self.poly_start_idxs[end_idx]
        else:
            next_idx = next_idx + 1

        i = 0
        while next_idx not in [cp.bs1_idx, cp.bs2_idx]:
            next_p = self.vor.points[next_idx]
            ring.append(next_p)

            # check if the next index is a critical point
            new_cp = False
            idx_cp = None
            if next_idx in self.cp_base1_idxs:
                idx_cp = self.cp_base1_idxs.index(next_idx)
                if not self.critical_points[idx_cp].skip:
                    ring.append(self.critical_points[idx_cp].vp.v)
                    ring.append(self.vor.points[self.cp_base2_idxs[idx_cp]])
                    next_idx = self.cp_base2_idxs[idx_cp]
                    self.critical_points[idx_cp].passed_12 = True
                    self.critical_points[idx_cp].n_passed += 1
                    new_cp = True
            elif next_idx in self.cp_base2_idxs:
                idx_cp = self.cp_base2_idxs.index(next_idx)
                if not self.critical_points[idx_cp].skip:
                    ring.append(self.critical_points[idx_cp].vp.v)
                    ring.append(self.vor.points[self.cp_base1_idxs[idx_cp]])
                    next_idx = self.cp_base1_idxs[idx_cp]
                    self.critical_points[idx_cp].passed_21 = True
                    self.critical_points[idx_cp].n_passed += 1
                    new_cp = True

            # check if the critical point is already in the ring
            if new_cp:
                new_cp_idx = idx_cp
                if new_cp_idx in ring_cp_idxs:
                    idx = ring_cp_idxs.index(new_cp_idx)
                    found_cp_idx = ring_cp_ring_idxs[idx]
                    # cut the ring at the found critical point
                    ring = ring[:found_cp_idx]
                    self.critical_points[idx_cp].skip = True
                    self.lg.warn(f"Critical point {idx_cp} already in the ring, skipping.")
                else:
                    ring_cp_idxs.append(new_cp_idx)
                    ring_cp_ring_idxs.append(len(ring) - 2)

            # check if reached end of the polygon
            if next_idx in self.poly_end_idxs:
                end_idx = self.poly_end_idxs.index(next_idx)
                next_idx = self.poly_start_idxs[end_idx]
            else:
                next_idx = next_idx + 1

            #
            i += 1
            if i > n_points:
                raise ValueError("Infinite loop detected in creating critical point segment.")

        # add the last point
        next_p = self.vor.points[next_idx]
        ring.append(next_p)
        ring.append(cp.vp.v)
        return np.array(ring), set(ring_cp_idxs)

    def create_cells(self):
        """Create cells from the rings."""
        self.lg.info(f"Creating cells from {len(self.rings)} rings.")
        for i, ring in enumerate(self.rings):
            if len(ring) < 3:
                self.lg.warn(f"Ring {i} has less than 3 points, skipping.")
                continue
            cell = Cell(ind=i, poly=Polygon(ring))
            c_crit_points = [self.critical_points[j] for j in self.ring_critp_dict.get(i, set())]
            cell.set_critical_points(c_crit_points)
            self.mcells.append(cell)

    def eval_cells(self) -> None:
        for cell in self.mcells:
            cell.eval_cell()

    def create_sequence(self, distances: np.ndarray = None) -> None:
        if len(self.mcells) == 0:
            return
        elif len(self.mcells) == 1:
            self.smcells = self.mcells
            return

        # create optimized sequence of cells
        opt_seq = OptimCellsSeq(mcells=self.mcells, zmd=self.zmd, distances=distances)
        self.seq_idxs = opt_seq.tour_idxs
        self.smcells = [self.mcells[i] for i in opt_seq.tour_idxs]

    def plot_rings(self) -> None:
        if not (self.dcfgs.plot_decomposition or self.dcfgs.save_decomposition_plot):
            self.lg.info("Skipping voronoi decomposition.")
            return
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_aspect('equal', adjustable='box')
        n_rings = len(self.rings)
        colors = plt.cm.viridis(np.linspace(0, 1, n_rings))
        for i, ring in enumerate(self.rings):
            ax.plot(ring[:, 0], ring[:, 1], color=colors[i, :], linewidth=1, label='Critical Segment')

        for i, cp in enumerate(self.critical_points):
            ax.plot(cp.vp.v[0], cp.vp.v[1], 'o', color='red', markersize=5)
            ax.annotate(f"{i}", (cp.vp.v[0], cp.vp.v[1]), fontsize=6, color='blue')
        #
        if self.dcfgs.save_decomposition_plot:
            save_path = self.zmd.result_path + "/rings.png"
            fig.savefig(save_path, dpi=300)

        #
        if self.dcfgs.plot_decomposition:
            plt.show()
