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


# Example usage (assuming samples is a numpy array shape (N, D) and m_true length D):
# samples = np.random.randn(10000, 5)
# m_true = np.array([0.5, 1.2, -0.3, 2.0, 0.7])
# labels = ["param1", "param2", "param3", "param4", "param5"]
# plot_marginals_and_pairs(samples, labels=labels, truths=m_true, bins=40, save_path="corner_plot.png")


import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def plot_5d_corner(samples, fontsize=20,
                   labels=None,
                   truths=None,
                   bins=80,
                   figsize=(12, 12),
                   save_path=None):
    samples = np.asarray(samples)
    N, D = samples.shape
    assert D == 5, "Need 5D samples"

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": fontsize,          # default, affects text (titles, legends, etc)
        "axes.labelsize": fontsize,     # for your G_frame, ρ, φ labels
        "xtick.labelsize": 4,    # smaller tick numbers on all axes
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
    
    # Your fixed parameter ranges:
    axis_ranges = [
        (0, 10),     # θ₀
        (0, 10),     # θ₁
        (7, 10),     # θ₂
        (0.3, 0.5),  # θ₃
        (42, 48),    # θ₄
    ]
    
    fig, axes = plt.subplots(D, D, figsize=figsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            
            # Upper triangle: blank
            if j > i:
                ax.axis("off")
                continue
            
            # Diagonal panels: 1D histograms
            if i == j:
                data = samples[:, i]
                # fix x-axis, let y-axis autoscale
                ax.set_xlim(axis_ranges[i])
                ax.hist(data,
                        bins=bins,
                        range=axis_ranges[i],
                        density=True,
                        histtype="stepfilled",
                        alpha=1,
                        color="midnightblue")
                if truths is not None:
                    ax.axvline(truths[i],
                               color="k",
                               linestyle="--",
                               lw=1)
                ax.set_yticks([])  # No y-ticks on marginals
            
            # Lower triangle: 2D density
            else:
                x, y = samples[:, j], samples[:, i]
                xr, yr = axis_ranges[j], axis_ranges[i]
                # fix both axes
                ax.set_xlim(xr)
                ax.set_ylim(yr)
                ax.hist2d(x,
                          y,
                          bins=bins,
                          range=[xr, yr],
                          cmap="viridis",
                          norm=LogNorm())
                if truths is not None:
                    ax.plot(truths[j],
                            truths[i],
                            "wx",
                            markersize=6,
                            markeredgewidth=2)
            
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