""" Plotting and saving figures of resulted cells. """
import matplotlib.pyplot as plt
from covpp.base.configs import PipeLineConfig
from covpp.segmentation.cell_planning import PCell


class CellsPlotter:
    """
    Plotting and saving figures of the resulted cells from segmentation and sequencing.

    Includes:
        - visualize_cells
        - visualize_sequence
    """

    def __init__(self, method_results_dir: str, segmentation_dir: str):
        self.method_results_dir = method_results_dir
        self.segmentation_dir = segmentation_dir

        # figures
        self.fig_mcells = None
        self.fig_mcells_seq = None

    def run(self, cells: list[PCell]):
        """ Main run function to plot and save figures based on the input cells. """
        self.fig_mcells, _ = self.visualize_cells(cells)
        self.visualize_cells_sequence(cells)

        # save figures
        if PipeLineConfig.do_plot_segmentation:
            self.save_figs()

    def visualize_cells_sequence(self, cells: list[PCell]):
        """ Visualize the sequence on cells figure. """
        fig_mcells, ax_mcells = self.visualize_cells(cells)
        self.visualize_sequence(cells, ax_mcells)
        self.fig_mcells_seq = fig_mcells

    def visualize_cells(self, cells: list[PCell]):
        """ Visualize cells obtained from segmentation. """

        fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(12, 8))
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

        ng = len(cells)
        colors = plt.cm.get_cmap('tab20', 2 * ng).colors
        for i, cell in enumerate(cells):
            if i % 2:
                color = colors[i]
            else:
                color = colors[i + ng]

            # cell polygon
            x, y = cell.poly.exterior.xy
            cx, cy = cell.cxy
            ax.plot(x, y, color=color, linewidth=2, linestyle='-')
            ax.fill(x, y, alpha=0.2, fc=color)
            ax.text(cx, cy, f"${cell.id}$", color='black', fontsize=10)

            # rectangle polygon
            if cell.rectangle_poly is not None:
                rx, ry = cell.rectangle_poly.exterior.xy
                ax.plot(rx, ry, color='black', linewidth=2, linestyle='--')

        # manually create legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='black', lw=2, linestyle='-', label='Cell Polygon'),
                           Line2D([0], [0], color='black', lw=2, linestyle='--', label='Inscribed Rectangle')]
        # ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=12,
        #           edgecolor='black',
        #           bbox_to_anchor=(1.1, 1.05),
        #           borderaxespad=0.0,)
        ax.invert_yaxis()
        return fig, ax

    def visualize_sequence(self, cells: list[PCell], ax):
        """ Visualize the sequencing path on the cells figure. """
        for i, cell in enumerate(cells):
            x1, y1 = cell.cxy
            # ax.text(x1, y1, f"${i}$", color='black', fontsize=8)

            # connection line
            next_ind = cell.next_cell_id
            if next_ind is None:
                continue
            x2, y2 = cells[next_ind].cxy
            ax.plot([x1, x2], [y1, y2], color="gray", linewidth=1, linestyle='--')

    def save_figs(self):
        """ Save the plotted figures to files. """
        self.fig_mcells.savefig(self.segmentation_dir + "/cells.png", format="png", dpi=PipeLineConfig.dpi)
        if self.fig_mcells_seq:
            self.fig_mcells_seq.savefig(self.method_results_dir + "/sequence.png", format="png", dpi=PipeLineConfig.dpi)
        plt.close('all')
