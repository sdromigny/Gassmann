import numpy as np
import matplotlib.pyplot as plt

def plot_marginals_and_pairs(samples, labels=None, truths=None, bins=50, figsize=None, save_path=None):
    """
    Create a corner plot showing 1D marginal PDFs on the diagonal and 2D histograms for joint PDFs in the lower triangle.
    
    Parameters:
    - samples: np.ndarray of shape (N_samples, D), MCMC samples.
    - labels: list of strings of length D for parameter names. If None, uses generic names.
    - truths: array-like of length D with true parameter values; plotted as vertical/horizontal lines or points.
    - bins: int or sequence, number of bins for histograms.
    - figsize: tuple, e.g. (3*D, 3*D) for size of figure; if None, defaults to (3*D, 3*D).
    - save_path: if provided, save the figure to this path.
    """
    samples = np.asarray(samples)
    N, D = samples.shape
    if labels is None:
        labels = [f"$\\theta_{{{i}}}$" for i in range(D)]
    if truths is not None:
        truths = np.asarray(truths)
        if truths.shape[0] != D:
            raise ValueError(f"Length of truths ({truths.shape[0]}) does not match number of parameters ({D}).")
    
    # Create subplots: D x D
    if figsize is None:
        figsize = (3 * D, 3 * D)
    fig, axes = plt.subplots(D, D, figsize=figsize)
    
    # Loop over grid
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            # Diagonal: 1D marginal for parameter i
            if i == j:
                ax.hist(samples[:, i], bins=bins, density=True, alpha=0.7)
                if truths is not None:
                    ax.axvline(truths[i], color='k', linestyle='--')
                ax.set_yticks([])
            # Lower triangle: joint 2D histogram of (param j on x-axis, param i on y-axis)
            elif j < i:
                # 2D histogram
                ax.hist2d(samples[:, j], samples[:, i], bins=bins, density=True)
                if truths is not None:
                    ax.plot(truths[j], truths[i], marker='x', color='w')
            # Upper triangle: turn off axis
            else:
                ax.axis('off')
            
            # Labeling: only bottom row gets x-labels, only left column gets y-labels
            if i == D - 1 and j < D:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticks([])
            if j == 0 and i < D:
                if i != j:
                    ax.set_ylabel(labels[i])
            else:
                ax.set_yticklabels([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_5d_corner(samples=None, fontsize=20,
                   labels=None,
                   truths=None,
                   bins=80,
                   figsize=(12, 12),
                   save_path=None):
    """
    Plot a 5D corner plot. If samples is None or empty (0 rows), the
    function will only draw axis frames and truth markers (if provided).
    """
    # Accept samples=None or an array with zero rows
    if samples is None:
        samples = np.empty((0, 5))
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != 5:
        raise AssertionError("Need samples with shape (N,5) or samples=None")
    N, D = samples.shape

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": 4,
        "ytick.labelsize": 4,
    })

    if labels is None:
        labels = [
            r"$\rho_{\rm fluid,1}$",
            r"$\rho_{\rm fluid,2}$",
            r"$G_{\rm frame}$",
            r"$\rho_{\rm grain,2}$",
            r"$\phi$",
        ]
    if truths is not None:
        truths = np.asarray(truths)
        assert truths.shape[0] == D

    # fixed parameter ranges:
    axis_ranges = [
        (0, 10),     # θ₀
        (0, 10),     # θ₁
        (7, 10),     # θ₂
        (0.3, 0.5),  # θ₃
        (42, 48),    # θ₄
    ]

    has_samples = N > 0

    fig, axes = plt.subplots(D, D, figsize=figsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]

            # Upper triangle: blank
            if j > i:
                ax.axis("off")
                continue

            # Diagonal panels: 1D histograms (or empty with vline)
            if i == j:
                ax.set_xlim(axis_ranges[i])
                if has_samples:
                    data = samples[:, i]
                    ax.hist(data,
                            bins=bins,
                            range=axis_ranges[i],
                            density=True,
                            histtype="stepfilled",
                            alpha=1,
                            color="midnightblue")
                else:
                    # keep the panel empty but set a reasonable y-limits
                    ax.set_ylim(0, 1)
                    ax.set_yticks([])

                if truths is not None:
                    ax.axvline(truths[i],
                               color="k",
                               linestyle="--",
                               lw=1)
                ax.set_yticks([])  # No y-ticks on marginals

            # Lower triangle: 2D density (or empty background)
            else:
                xr, yr = axis_ranges[j], axis_ranges[i]
                ax.set_xlim(xr)
                ax.set_ylim(yr)
                if has_samples:
                    x, y = samples[:, j], samples[:, i]
                    ax.hist2d(x,
                              y,
                              bins=bins,
                              range=[xr, yr],
                              cmap="viridis",
                              norm=LogNorm())
                # else: leave blank axes (no points/hist2d)
                if truths is not None:
                    ax.scatter([truths[j]],
                               [truths[i]],
                               s=100,
                               marker="o",
                               facecolors="red",
                               edgecolors="red",
                               linewidths=0.8,
                               zorder=10)

            # Axis labeling only on bottom row / left column
            if i == D - 1:
                ax.set_xlabel(labels[j], fontsize=fontsize)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(labels[i], fontsize=fontsize)
            else:
                ax.set_yticks([])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
