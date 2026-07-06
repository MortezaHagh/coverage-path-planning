""" Plot styles for CovPP visualizations."""
import scienceplots  # pylint: disable=unused-import
import matplotlib.pyplot as plt
from covpp.base.configs import PipeLineConfig


def configure_ax(ax, title, show_legend=True, show_title=True):
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if show_title:
        ax.set_title(title, fontsize='medium')
    if show_legend:
        ax.legend(
            frameon=True,
            edgecolor='black',
            fontsize=10,
            loc='upper left',
            bbox_to_anchor=(1.0, 1),
            borderaxespad=0.0,
        )


def set_plot_styles(fig_w=None, fig_h=None) -> None:
    """ Set plot styles for CovPP visualizations. """
    # professional styles
    plt.style.use(['science'])
    plt.rcParams['figure.dpi'] = PipeLineConfig.dpi
    # Avoid LaTeX dependency in environments without full TeX install
    plt.rcParams['text.usetex'] = False
    if fig_w and fig_h:
        plt.rcParams['figure.figsize'] = (fig_w, fig_h)

# plt.rcParams['figure.constrained_layout.use'] = True
# fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)

# ======================================================================================================================
# visualization styles
# ======================================================================================================================

# plt.rc('font', family='serif')
# plt.rc('xtick', labelsize='x-small')
# plt.rc('ytick', labelsize='x-small')


# ------------------------------------------------------------------------------


# import scienceplots

# # Use the science style
# plt.style.use(['science'])

# ------------------------------------------------------------------------------

# # Use IEEE style with grid
# plt.style.use(['science', 'ieee', 'grid'])

# ------------------------------------------------------------------------------

# import matplotlib_inline
# matplotlib_inline.backend_inline.set_matplotlib_formats("svg")

# plt.rcParams.update({
#     "figure.figsize": (9, 6),
#     "axes.spines.top": False,
#     "axes.spines.right": False,
#     "font.size": 14,
#     "figure.titlesize": "xx-large",
#     "xtick.labelsize": "medium",
#     "ytick.labelsize": "medium",
#     "axes.axisbelow": True
# })

# ------------------------------------------------------------------------------

# # PLT PARAMS
# # plt.rcParams['figure.constrained_layout.use'] = True
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.size'] = 10
# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['figure.figsize'] = (15, 10)
# plt.rcParams['text.usetex'] = True  # TeX rendering
