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
