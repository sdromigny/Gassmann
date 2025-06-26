import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def pairplot(generated_data_2d,
                      m_true,
                      fontsize=14,
                      bins=50,
                      figsize=(7, 7),
                      cmap="viridis",
                      save_path=None):
    """
    2D histogram + detached marginals:
     - main panel bottom‐left, ticks only on left & bottom
     - x‐marginal above, showing density ticks on its y‐axis
     - y‐marginal right, showing density ticks on its x‐axis
    """

    plt.rcParams.update({
        "font.family": "serif",   # Use serif font
        "font.size": fontsize,          # General font size
        "axes.labelsize": fontsize,     # Font size for axis labels
        "xtick.labelsize": 15,    # Font size for x-axis ticks
        "ytick.labelsize":15,    # Font size for y-axis ticks
        "legend.fontsize": fontsize,    # Font size for legends
        "figure.titlesize": fontsize    # Font size for figure titles
    })
    x = generated_data_2d[:, 0]
    y = generated_data_2d[:, 1]

    x_lo, x_hi = 0.0, 10.0
    y_lo, y_hi = 0.0, 10.0

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1, 4],
        width_ratios=[4, 1],
        hspace=0.1, wspace=0.1
    )

    # — Top marginal (x) —
    ax_x = fig.add_subplot(gs[0, 0])
    counts_x, edges_x = np.histogram(
        x,
        bins=np.linspace(x_lo, x_hi, bins + 1),
        density=True
    )
    centers_x = 0.5 * (edges_x[:-1] + edges_x[1:])
    ax_x.fill_between(
        centers_x,
        counts_x,
        step="mid",
        alpha=1,
        color="midnightblue"
    )
    ax_x.axvline(m_true[0], color="red", linestyle="--")
    ax_x.set_xlim(x_lo, x_hi)
    ax_x.set_ylim(0, counts_x.max() * 1.05)
    ax_x.set_ylabel("Density", fontsize=fontsize - 2)
    # show y‐axis ticks, hide x‐axis ticks
    ax_x.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_x.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, labelsize=fontsize-4)
    ax_x.spines['bottom'].set_visible(False)

    # — Main 2D histogram —
    ax_main = fig.add_subplot(gs[1, 0])
    h2d = ax_main.hist2d(
        x, y,
        bins=(bins, bins),
        range=((x_lo, x_hi), (y_lo, y_hi)),
        cmap=cmap,
        norm=LogNorm()
    )
    #ax_main.scatter(x, y, s=5, c="white", alpha=0.6, linewidths=0)
    ax_main.plot(m_true[0], m_true[1], 'ro', markersize=6)
    ax_main.set_xlim(x_lo, x_hi)
    ax_main.set_ylim(y_lo, y_hi)
    ax_main.set_xlabel(r"$\rho_{\rm fluid,1}$", fontsize=fontsize)
    ax_main.set_ylabel(r"$\rho_{\rm fluid,2}$", fontsize=fontsize)
    # ticks only on bottom & left
    ax_main.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labelsize=fontsize-2)
    ax_main.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, labelsize=fontsize-2)

    # — Right marginal (y) —
    ax_y = fig.add_subplot(gs[1, 1])
    counts_y, edges_y = np.histogram(
        y,
        bins=np.linspace(y_lo, y_hi, bins + 1),
        density=True
    )
    centers_y = 0.5 * (edges_y[:-1] + edges_y[1:])
    ax_y.fill_betweenx(
        centers_y,
        counts_y,
        step="mid",
        alpha=1,
        color="midnightblue"
    )
    ax_y.axhline(m_true[1], color="red", linestyle="--")
    ax_y.set_ylim(y_lo, y_hi)
    ax_y.set_xlim(0, counts_y.max() * 1.05)
    ax_y.set_xlabel("Density", fontsize=fontsize - 2)
    # show x‐axis ticks, hide y‐axis ticks
    ax_y.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_y.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labelsize=fontsize-4)
    ax_y.spines['left'].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig
