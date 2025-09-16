import matplotlib.pyplot as plt
from scripts.base.configs import DistrictConfigs
from scripts.planning.boustrophedon.cell import Cell, CellGroup


class BoustPlotter:
    """Boustrophedon decomposition plotter. 
        plots: boust cells and mcells. 
    """

    def __init__(self, dec_obj):
        self.dec_obj = dec_obj
        self.dcfgs = DistrictConfigs()
        self.result_path: str = dec_obj.result_path
        #
        self.viz_together = True
        self.set_aspect = True
        #
        self.ax_cell_event = None
        self.fig_cells_events = None
        self.fig_mcells = None
        self.fig_mcells_seq = None
        self.fig_all = None

    def visualize(self):
        if not (self.dcfgs.save_decomposition_plot or self.dcfgs.plot_decomposition):
            return
        if self.viz_together:
            self.visualize_together()
        else:
            self.visualize_seperate()
        if self.dcfgs.plot_decomposition:
            plt.show()

        # save figures
        if self.dcfgs.save_decomposition_plot:
            self.save_figs()

    def visualize_seperate(self):
        self.vizualize_cells_events()
        self.visualize_cells_groups()

    def visualize_together(self):
        fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 10))
        if self.set_aspect:
            axs[0].set_aspect('equal', adjustable='box')
            axs[1].set_aspect('equal', adjustable='box')
        self.vizualize_cells_events(axs[0])
        self.visualize_cells_groups(axs[1])
        self.fig_all = fig

    def vizualize_cells_events(self, ax=None):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            if self.set_aspect:
                ax.set_aspect('equal', adjustable='box')
        self.ax_cell_event = ax

        # # vizualize events
        # self.visualize_events()

        # plot contours
        self.plot_contours()

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
        for e in self.dec_obj.events:
            text1 = e.name
            self.ax_cell_event.text(e.xy[0], e.xy[1], text1, fontsize=10, color='blue')

    def plot_contours(self):
        for obs in self.dec_obj.obstacles:
            if obs.ind == 0:
                color = 'blue'
            else:
                color = 'red'
            x, y = obs.poly.exterior.xy
            self.ax_cell_event.plot(x, y, color=color, linewidth=1)

    def visualize_cells_groups(self, ax=None):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            if self.set_aspect:
                ax.set_aspect('equal', adjustable='box')
        ng = len(self.dec_obj.cells_groups)
        cg: CellGroup
        colors = plt.cm.get_cmap('tab20', 2*ng).colors
        for i, cg in enumerate(self.dec_obj.cells_groups):
            if i % 2:
                color = colors[i]
            else:
                color = colors[i+ng]

            # merged cell
            x, y = cg.merged_cell.poly.exterior.xy
            cx, cy = cg.merged_cell.cxy
            text = f"g{i}"
            ax.text(cx, cy, text, color='black', fontsize=8)
            ax.plot(x, y, color=color, linewidth=4, linestyle='-')
            ax.plot(cx, cy, 'o', color='purple', markersize=6)
            #
            for cell in cg.cells:
                # fill
                x, y = cell.poly.exterior.xy
                ax.fill(x, y, alpha=0.2, fc=color, linewidth=1, linestyle='--', edgecolor='gray')

            # adjacents
            # m1
            mc: Cell = cg.merged_cell
            mcx, mcy = mc.cxy
            for ac_i in mc.mid_adjlist:
                acx, acy = self.dec_obj.mcells[ac_i].cxy
                ax.plot([mcx, acx], [mcy, acy], color=color, linewidth=1, linestyle='--')

        #
        if fig:
            self.fig_mcells = fig

    def plot_adj_gcells(self, ax, cell: Cell, adj_g_cell_ind: int, color='red'):
        c = cell.poly.centroid
        adj_c = self.dec_obj.cells_groups[adj_g_cell_ind].merged_cell.poly.centroid
        ax.plot([c.x, adj_c.x], [c.y, adj_c.y], color=color, linewidth=1, linestyle='--')

    def save_figs(self):
        ns = "/"
        dpi = 300
        path = self.result_path
        if self.viz_together:
            self.fig_all.savefig(path + f"{ns}boust_decomposition.png", format="png", dpi=dpi)
        else:
            self.fig_mcells.savefig(path + f"{ns}mcells.png", format="png", dpi=dpi)
            self.fig_cells_events.savefig(path + f"{ns}cells.png", format="png", dpi=dpi)
