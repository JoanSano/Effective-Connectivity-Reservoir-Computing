import matplotlib.pylab as plt
import numpy as np

def plot_RCC_Evidence(lags, *to_plot, **kwargs):
    """
    TODO: Add description

    Arguments
    ---------
    lags:
    *to_plot: (dicts) where the keys ["data", "error", "label", "color", "style", "linewidth"]
    """

    # Location of the maximum correlation
    if 'scale' in kwargs.keys():
        lags = lags * kwargs['scale'] # Repetition Time (TR)

    # Instantiate figure
    fig, ax = plt.subplots(figsize=(6,4))
    ax.remove()

    ##################
    left, bottom, width, height = [0.1, 0.12 , 0.85, 0.85]
    ax1 = fig.add_axes([left, bottom, width, height])
    # Plot main curves
    for curve in to_plot:    
        ax1.plot(lags, curve["data"], curve["linewidth"], color=curve["color"], linestyle=curve["style"], label=curve["label"], alpha=curve["alpha"])
        ax1.fill_between(lags, curve["data"]-curve["error"], curve["data"]+curve["error"],
            alpha=0.2, edgecolor=curve["color"], facecolor=curve["color"], linewidth=1, label=""        
        )
    # Plot significant times: It requires 
    if "significance_marks" in kwargs.keys():
        y_ini = -0.01
        for curve in kwargs["significance_marks"]:
            ax1.fill_between(curve["data"]*lags, y_ini, y_ini+0.02, alpha=0.8, facecolor=curve["color"], linewidth=0, label=curve["label"])
            y_ini += 0.02
    # Figures details
    z_min, z_max = kwargs["limits"]
    ax1.vlines(x=0, ymin=z_min, ymax=z_max, linewidth=0.3, color='grey', linestyles='--')
    ax1.hlines(y=0, xmin=lags[0], xmax=lags[-1], linewidth=0.3, color='grey', linestyles='--')
    ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)
    ax1.set_ylabel(kwargs['y_label'], fontsize=15)
    ax1.set_yticks([z_min,z_max]), ax1.set_yticklabels([str(z_min), str(z_max)]), ax1.set_ylim([z_min-0.01,z_max+0.02])
    ax1.set_xlim([lags[0],lags[-1]]), ax1.set_xlabel(kwargs['x_label'], fontsize=15)
    # Legend
    plt.legend(fontsize=8, frameon=False, ncols=2)
    
    ###############
    # Visualization
    if 'save' in kwargs.keys():
        format = kwargs['save'].split(".")[-1]
        if 'dpi' in kwargs.keys():
            plt.savefig(kwargs['save'], dpi=kwargs['dpi'], format=format)
        else:
            plt.savefig(kwargs['save'], format=format)
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':
    pass