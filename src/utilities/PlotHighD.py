


import numpy as np
import matplotlib.pyplot as plt
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
        "font.size": fontsize,          
        "axes.labelsize": fontsize,     
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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
    

    axis_ranges = [
        (0, 10),    
        (0, 10),    
        (7, 10),     
        (0.3, 0.5),  
        (42, 48),   
    ]
    
    fig, axes = plt.subplots(D, D, figsize=figsize)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)


    vmin, vmax = 1e-7, 1e+1
    
    img = None
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            

            if j > i:
                ax.axis("off")
                continue

            if i == j:
                data = samples[:, i]
                ax.set_xlim(axis_ranges[i])
                ax.hist(data,
                        bins=bins,
                        range=axis_ranges[i],
                        density=True,
                        histtype="stepfilled",
                        alpha=1,
                        color="midnightblue")
                if truths is not None:
                    # red dashed vertical line at truth
                    ax.axvline(truths[i],
                               color="red",
                               linestyle="--", 
                               lw=2)
                ax.set_yticks([])  
            else:
                x, y = samples[:, j], samples[:, i]
                xr, yr = axis_ranges[j], axis_ranges[i]
                ax.set_xlim(xr)
                ax.set_ylim(yr)


                H, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                                range=[xr, yr],
                                                density=True)

                X, Y = np.meshgrid(xedges, yedges)
                img = ax.pcolormesh(X, Y, H.T, cmap="viridis", norm=LogNorm(vmin=vmin, vmax=vmax))


                if truths is not None:
                    ax.plot(truths[j], truths[i],
                            marker='o',
                            color='red',
                            markersize=15,
                            markeredgecolor='white',
                            markeredgewidth=4,
                            zorder=10)
            

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
