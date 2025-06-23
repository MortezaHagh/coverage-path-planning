import matplotlib.pyplot as plt
from scripts.planning.boustrophedon.cell import Cell


class PlannerPlotter:
    def __init__(self, result_path: str):
        self.result_path = result_path
        self.fig_mcells = None
        self.fig_mcells_seq = None

    def run(self, mcells: list[Cell]):
        fig_mcells, _ = self.visualize_mcells(mcells)
        self.fig_mcells = fig_mcells
        #
        self.visualize_cells_sequence(mcells)

        # todo
        # if not self.dcfgs.save_decomposition_plot:
        self.save_figs()

    def visualize_cells_sequence(self, mcells: list[Cell]):
        fig_mcells, ax_mcells = self.visualize_mcells(mcells)
        self.visualize_sequence(mcells, ax_mcells)
        self.fig_mcells_seq = fig_mcells

    def visualize_mcells(self, mcells: list[Cell]):
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ng = len(mcells)
        colors = plt.cm.get_cmap('tab20', 2*ng).colors
        for i, cell in enumerate(mcells):
            if i % 2:
                color = colors[i]
            else:
                color = colors[i+ng]

            # merged cell
            x, y = cell.poly.exterior.xy
            # cx, cy = cell.cxy
            # text = f"g{i}-({cell.opt_id})"
            # ax.text(cx, cy, text, color='black', fontsize=8)
            # ax.plot(cx, cy, 'o', color='purple', markersize=6)
            ax.plot(x, y, color=color, linewidth=4, linestyle='-')
            ax.fill(x, y, alpha=0.2, fc=color)
        return fig, ax

    def visualize_sequence(self, mcells: list[Cell], ax):
        for i, cell in enumerate(mcells):
            next_ind = cell.next_cell_seq
            if next_ind is None:
                continue
            x1, y1 = cell.cxy
            x2, y2 = mcells[next_ind].cxy
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1, linestyle='--')
            #
            # text = f"g{i}-({cell.opt_id})"
            text = f"{i}"
            ax.text(x1, y1, text, color='black', fontsize=8)

    def save_figs(self):
        dpi = 300
        self.fig_mcells.savefig(self.result_path + "/mcells.png", format="png", dpi=dpi)
        self.fig_mcells_seq.savefig(self.result_path + "/sequence.png", format="png", dpi=dpi)
