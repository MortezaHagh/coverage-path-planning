import matplotlib.pyplot as plt
from scripts.decomposition.cell import Cell, CellGroup
from scripts.base.configs import DistrictConfgs


class DecPlotter:
    def __init__(self, dec_obj):
        self.dec_obj = dec_obj
        self.dcfgs = DistrictConfgs()
        #
        self.ax1 = None
        self.fig1 = None
        self.fig2 = None

    def visualize(self):
        if not self.dcfgs.save_decomposition_plot:
            return
        #
        fig, ax = plt.subplots(1, 1)
        self.fig1 = fig
        self.ax1 = ax
        ax.set_aspect('equal', adjustable='box')

        # plot contours
        self.plot_contours()

        # # vizualize events
        # self.visualize_events()

        # plot trapezoids
        self.vizualize_cells()

        # vizualize cells groups
        self.visualize_cells_groups()

        #
        if self.dcfgs.plot_decomposition:
            plt.show()

    def plot_contours(self):
        # plot contours
        for obs in self.dec_obj.obstacles:
            if obs.ind == 0:
                color = 'blue'
            else:
                color = 'red'
            x, y = obs.poly.exterior.xy
            self.ax1.plot(x, y, color=color, linewidth=1)

    def visualize_events(self):
        for e in self.dec_obj.events:
            # ct = "concave" if e.concave else "convex"
            # text1 = f"{e.sd_ind}-[t1:{e.etype}]-[t2:{e.etype_2}]--{ct}"
            # #
            # text2 = f"{e.ind}--[{e.prev_ind},{e.next_ind}]"
            # plt.text(e.xy[0], e.xy[1]-5, text2, fontsize=12, color='green')

            # name
            text1 = e.name
            self.ax1.text(e.xy[0], e.xy[1], text1, fontsize=8, color='blue')

            # #
            # # plt.plot(e.xy[0], e.xy[1], 'o', color='purple', markersize=8)
            # if e.hit_down_event is not None:
            #     # plt.plot(e.xy[0], e.hit_down_y, 'o', color='black', markersize=5)
            #     self.ax1.plot([e.xy[0], e.xy[0]], [e.xy[1], e.hit_down_y], '--', color='orange', linewidth=1)

    def vizualize_cells(self):
        for cell in self.dec_obj.cells:
            color = 'yellow'
            if not cell.status:
                color = 'red'

            # fill
            x, y = cell.poly.exterior.xy
            self.ax1.fill(x, y, alpha=0.5, fc=color, ec='black', linewidth=1)

            # centroid
            self.ax1.plot(cell.poly.centroid.x, cell.poly.centroid.y, 'o', color='purple', markersize=6)

            # # text
            # self.ax1.text(cell.poly.centroid.x, cell.poly.centroid.y, f"[{cell.ind}]", fontsize=10, color='black')
            # # self.ax1.text(cell.poly.centroid.x, cell.poly.centroid.y,
            # #              f"[{cell.ind}][{cell.down_adj_ci}, {cell.up_adj_ci}]", fontsize=10, color='black')

            # adjacents
            for mid_adj in cell.mid_adjlist:
                self.ax1.plot([cell.poly.centroid.x, self.dec_obj.cells[mid_adj].poly.centroid.x],
                              [cell.poly.centroid.y, self.dec_obj.cells[mid_adj].poly.centroid.y], 'b--', linewidth=1)
            if cell.up_adj_ci:
                self.ax1.plot([cell.poly.centroid.x, self.dec_obj.cells[cell.up_adj_ci].poly.centroid.x],
                              [cell.poly.centroid.y, self.dec_obj.cells[cell.up_adj_ci].poly.centroid.y], 'r--', linewidth=1)
            if cell.down_adj_ci:
                self.ax1.plot([cell.poly.centroid.x, self.dec_obj.cells[cell.down_adj_ci].poly.centroid.x],
                              [cell.poly.centroid.y, self.dec_obj.cells[cell.down_adj_ci].poly.centroid.y], 'g--', linewidth=1)

    def visualize_cells_groups(self):
        fig, ax = plt.subplots(1, 1)
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
            ax.text(cx, cy, f'g{i}', color='black', fontsize=10)
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
            for ac_i in mc.mid_adjlist:
                mcx, mcy = mc.cxy
                acx, acy = self.dec_obj.mcells[ac_i].cxy
                ax.plot([mcx, acx], [mcy, acy], color=color, linewidth=1, linestyle='--')

            # # m2
            # lcell = cg.last_cell
            # adj_cell: Cell = None
            # if lcell.up_adj_ci:
            #     adj_cell = self.dec_obj.cells[lcell.up_adj_ci]
            #     adj_g_cell_ind = adj_cell.group_ind
            #     self.plot_adj_gcells(ax, cg.merged_cell, adj_g_cell_ind, color='red')
            # if lcell.down_adj_ci:
            #     adj_cell = self.dec_obj.cells[lcell.down_adj_ci]
            #     adj_g_cell_ind = adj_cell.group_ind
            #     self.plot_adj_gcells(ax, cg.merged_cell, adj_g_cell_ind, color='green')
            # if lcell.mid_adjlist:
            #     for ac in lcell.mid_adjlist:
            #         adj_cell = self.dec_obj.cells[ac]
            #         adj_g_cell_ind = adj_cell.group_ind
            #         self.plot_adj_gcells(ax, cg.merged_cell, adj_g_cell_ind, color='blue')
        #
        self.fig2 = fig

    def plot_adj_gcells(self, ax, cell: Cell, adj_g_cell_ind: int, color='red'):
        c = cell.poly.centroid
        adj_c = self.dec_obj.cells_groups[adj_g_cell_ind].merged_cell.poly.centroid
        ax.plot([c.x, adj_c.x], [c.y, adj_c.y], color=color, linewidth=1, linestyle='--')

    def save_figs(self, path: str):
        self.fig1.savefig(path + "_1.png", format="png", dpi=400)
        self.fig2.savefig(path + "_2.png", format="png", dpi=400)
