""" Boustrophedon decomposition plotter """
import os
import matplotlib.pyplot as plt
from covpp.base.configs import PipeLineConfig
from covpp.coverage_planning.boustrophedon.cell import Cell, CellGroup
from covpp.coverage_planning.boustrophedon.decomposition_bs_mh import CellDecomposition


class BoustPlotter:
    """
    Boustrophedon decomposition plotter.
    Plots: boust cells and mcells.
    """

    def __init__(self):
        self.dec_obj: CellDecomposition = None
        self.results_dir: str = None

        self.viz_together = False

        #
        self.ax_cell_event = None
        self.fig_cells_events = None
        self.fig_mcells = None
        self.fig_mcells_seq = None
        self.fig_all = None

    def visualize(self, dec_obj: CellDecomposition):
        """ Visualize decomposition results """

        if not PipeLineConfig.do_plot_boust_decomposition:
            return

        self.dec_obj = dec_obj
        self.results_dir = dec_obj.results_dir

        if self.viz_together:
            self.visualize_together()
        else:
            self.visualize_seperate()

        self.save_figs()
        plt.close('all')

    def visualize_seperate(self):
        """ Plot cells/events and mcells/groups separately """
        self.vizualize_cells_events()
        self.visualize_cells_groups()

    def visualize_together(self):
        """ Plot both cells/events and mcells/groups together """
        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 10))
        axs[0].set_aspect('equal', adjustable='box')
        axs[1].set_aspect('equal', adjustable='box')
        axs[0].axis('off')
        axs[1].axis('off')

        self.vizualize_cells_events(axs[0])
        self.visualize_cells_groups(axs[1])
        self.fig_all = fig

    def vizualize_cells_events(self, ax=None):
        """ Plot cells and events """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.axis('off')
            ax.set_aspect('equal', adjustable='box')
        self.ax_cell_event = ax

        # # vizualize events
        # self.visualize_events()

        # plot contours
        self.plot_contours(ax=self.ax_cell_event)

        # plot cells
        for cell in self.dec_obj.cells:
            color = 'yellow'
            if not cell.found_connection:
                color = 'red'

            # fill
            x, y = cell.poly.exterior.xy
            self.ax_cell_event.fill(x, y, alpha=0.5, fc=color, ec='black', linewidth=1)

            # centroid
            self.ax_cell_event.plot(cell.poly.centroid.x, cell.poly.centroid.y, 'o', color='purple', markersize=6)

            # adjacents
            for mid_adj in cell.mid_adjlist:
                self.ax_cell_event.plot([cell.poly.centroid.x, self.dec_obj.cells[mid_adj].poly.centroid.x],
                                        [cell.poly.centroid.y, self.dec_obj.cells[mid_adj].poly.centroid.y], 'b--', linewidth=1)
            if cell.up_adj_ci:
                self.ax_cell_event.plot([cell.poly.centroid.x, self.dec_obj.cells[cell.up_adj_ci].poly.centroid.x],
                                        [cell.poly.centroid.y, self.dec_obj.cells[cell.up_adj_ci].poly.centroid.y], 'r--', linewidth=1)
            if cell.down_adj_ci:
                self.ax_cell_event.plot([cell.poly.centroid.x, self.dec_obj.cells[cell.down_adj_ci].poly.centroid.x],
                                        [cell.poly.centroid.y, self.dec_obj.cells[cell.down_adj_ci].poly.centroid.y], 'g--', linewidth=1)
        #
        if fig:
            self.fig_cells_events = fig

    def visualize_events(self):
        """ Plot events """
        for e in self.dec_obj.events:
            text1 = e.name
            self.ax_cell_event.text(e.xy[0], e.xy[1], text1, fontsize=10, color='blue')

    def plot_contours(self, ax):
        """ Plot obstacles contours """
        for obs in self.dec_obj.obstacles:
            x, y = obs.poly.exterior.xy
            if obs.ind == 0:
                color = 'blue'
            else:
                color = 'red'
                # fill
                ax.fill(x, y, alpha=0.5, fc='gray')
            ax.plot(x, y, color=color, linewidth=1)

    def visualize_cells_groups(self, ax=None):
        """ Plot merged cells and their groups """
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1, constrained_layout=True)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')

        # plot contours
        self.plot_contours(ax=ax)

        ng = len(self.dec_obj.cells_groups)
        cg: CellGroup
        colors = plt.cm.get_cmap('tab20', 2 * ng).colors
        for i, cg in enumerate(self.dec_obj.cells_groups):
            if i % 2:
                color = colors[i]
            else:
                color = colors[i + ng]

            # merged cell
            merged_poly = cg.merged_cell.poly

            # # buffer to shrink the polygon for better visualization
            # dist = -6  # negative for shrinking
            # merged_poly = merged_poly.buffer(dist)

            x, y = merged_poly.exterior.xy
            cx, cy = cg.merged_cell.cxy
            text = f"g{i}"
            # ax.text(cx, cy, text, color='black', fontsize=8)
            ax.plot(cx, cy, 'o', color='purple', markersize=6)
            ax.plot(x, y, color=color, linewidth=2, linestyle='-')
            ax.fill(x, y, alpha=0.2, fc=color, linewidth=1, linestyle='--', edgecolor='gray')

            #
            # for cell in cg.cells:
            #     # fill
            #     x, y = cell.poly.exterior.xy
            #     ax.fill(x, y, alpha=0.2, fc=color, linewidth=1, linestyle='--', edgecolor='gray')

            # adjacents
            # m1
            mc: Cell = cg.merged_cell
            mcx, mcy = mc.cxy
            for ac_i in mc.mid_adjlist:
                acx, acy = self.dec_obj.mcells[ac_i].cxy
                ax.plot([mcx, acx], [mcy, acy], color=color, linewidth=1, linestyle='--')

        #
        ax.invert_yaxis()
        if fig:
            self.fig_mcells = fig

    def plot_adj_gcells(self, ax, cell: Cell, adj_g_cell_ind: int, color='red'):
        """ Plot adjacency between two mcells """
        c = cell.poly.centroid
        adj_c = self.dec_obj.cells_groups[adj_g_cell_ind].merged_cell.poly.centroid
        ax.plot([c.x, adj_c.x], [c.y, adj_c.y], color=color, linewidth=1, linestyle='--')

    def save_figs(self):
        """ save figures to results directory """
        ns = "/"
        cid = self.dec_obj.main_cell_id
        path = os.path.join(self.results_dir, "decomposition")
        os.makedirs(path, exist_ok=True)
        if self.viz_together:
            self.fig_all.savefig(path + f"{ns}bd_{cid}.png", format="png", dpi=PipeLineConfig.dpi)
        else:
            self.fig_mcells.savefig(path + f"{ns}mcells_{cid}.png", format="png", dpi=PipeLineConfig.dpi)
            self.fig_cells_events.savefig(path + f"{ns}cells_{cid}.png", format="png", dpi=PipeLineConfig.dpi)
        plt.close('all')
