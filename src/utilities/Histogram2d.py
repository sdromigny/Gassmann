
import numpy as np
import matplotlib.pyplot as plt
import os
#import wandb



def pairplot(generated_data_2d, m_true,fontsize, save_path=None):
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all figures (if needed)

    plt.rcParams.update({
        "font.family": "serif",   # Use serif font
        "font.size": fontsize,          # General font size
        "axes.labelsize": fontsize,     # Font size for axis labels
        "xtick.labelsize": 15,    # Font size for x-axis ticks
        "ytick.labelsize":15,    # Font size for y-axis ticks
        "legend.fontsize": fontsize,    # Font size for legends
        "figure.titlesize": fontsize    # Font size for figure titles
    })
    x_edges = np.linspace(0, 10, 100)  # 100 bins between 0 and 10
    y_edges = np.linspace(0, 10, 100)  # 100 bins between 0 and 10

    # x_edges = np.linspace(0, 2300, 100)  # 100 bins between 0 and 10
    # y_edges = np.linspace(0, 2300, 100)  # 100 bins between 0 and 10

    # Compute the 2D histogram (Z represents the density)
    Z, x_edges, y_edges = np.histogram2d(generated_data_2d[:, 0], generated_data_2d[:, 1], bins=(x_edges,y_edges), density=True)

    # To make the Z values represent densities (probability density), normalize by the total number of samples
    Z = Z / Z.sum()

    # Marginal distribution for X
    x_hist, _ = np.histogram(generated_data_2d[:, 0], bins=x_edges, density=True)

    # Marginal distribution for Y
    y_hist, _ = np.histogram(generated_data_2d[:, 1], bins=y_edges, density=True)


    # Plotting
    fig = plt.figure(figsize=(8, 8))
    grid = plt.GridSpec(4, 4, hspace=0.3, wspace=0.3)

    # Main 2D PDF plot
    ax_main = fig.add_subplot(grid[1:, :-1])

    # Contour plot using the 2D histogram
    X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])  # Meshgrid for contour plotting
    ax_main.contourf(X, Y, Z.T, levels=20, cmap="viridis")  # Transpose Z to match the shape of X, Y

    # Plot the m_true point as a red dot in the 2D histogram
    ax_main.plot(m_true[0], m_true[1], 'ro', label=r'true $\rho_{fluid}$ value')

    ax_main.set_xlabel(r"Posterior values of $\rho_{fluid,1}$",fontsize=fontsize)
    ax_main.set_ylabel(r"Posterior values of $\rho_{fluid,2}$", fontsize=fontsize)
    #ax_main.legend(fontsize=20)

    # Marginal PDF for X (top plot)
    ax_marg_x = fig.add_subplot(grid[0, :-1], sharex=ax_main)
    ax_marg_x.plot(x_edges[:-1], x_hist, color="blue")
    ax_marg_x.set_ylabel("Density",fontsize=fontsize)

    # Plot the m_true vertical line for X in the marginal plot
    ax_marg_x.axvline(m_true[0], color='red', linestyle='--', label=r'true $\rho_{fluid,1}$ value')
    #ax_marg_x.legend()

    # Marginal PDF for Y (right plot)
    ax_marg_y = fig.add_subplot(grid[1:, -1], sharey=ax_main)
    ax_marg_y.plot(y_hist, y_edges[:-1], color="blue")
    ax_marg_y.set_xlabel("Density",fontsize=fontsize)

    # Plot the m_true vertical line for Y in the marginal plot
    ax_marg_y.axhline(m_true[1], color='red', linestyle='--', label=r'$true $\rho_{fluid,2}$ value')
    #ax_marg_y.legend()

    # Log the figure with wandb (if needed)
    #toyflow_instance.logger.experiment.log({"2D histogram comparison": wandb.Image(fig)})
        # ✅ Save figure if save_path is provided
    if save_path:
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Figure saved to: {save_path}")

    return fig
    # Show the plot

