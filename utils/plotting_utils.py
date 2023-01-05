import matplotlib.pylab as plt
import numpy as np

def plot_RCC_input2output(lags, rho_i2o, rho_o2i, **kwargs):
    """
    TODO: Add description
    """

    if 'series_names' in kwargs.keys():
        X, Y = kwargs['series_names']
    else:
        X, Y = 'X', 'Y'

    # Location of the maximum correlation
    if 'scale' in kwargs.keys():
        lags = lags * kwargs['scale'] # Repetition Time (TR)
    max_i2o, max_o2i = lags[np.argmax(rho_o2i)], lags[np.argmax(rho_i2o)]
    max_rho_i2o, max_rho_o2i = np.max(rho_o2i), np.max(rho_i2o)

    # Instantiate figure
    fig, ax = plt.subplots(figsize=(6,4))
    ax.remove()

    ##################
    left, bottom, width, height = [0.1, 0.12 , 0.85, 0.85]
    ax1 = fig.add_axes([left, bottom, width, height])
    # Plot main curves
    ax1.plot(lags, rho_i2o, linewidth=2, color='blue', label=r'$\rho$'+f"[{X},{Y}]    "+X+r'$_{t}$'+' predicts '+Y+r'$_{t+\tau}$')
    ax1.plot(lags, rho_o2i, linewidth=2, color='red',  label=r'$\rho$'+f"[{Y},{X}]    "+Y+r'$_{t}$'+' predicts '+X+r'$_{t+\tau}$')
    ax1.vlines(x=0, ymin=max_rho_i2o-0.2, ymax=max_rho_i2o+0.05, linewidth=1, color='black', linestyles='--')
    # Display statistic errors (if present)
    if 'error_i2o' in kwargs.keys():
        ax1.fill_between(lags, rho_i2o-kwargs['error_i2o'], rho_i2o+kwargs['error_i2o'],
            alpha=0.2, edgecolor='blue', facecolor='blue',linewidth=1            
        )
    if 'error_o2i' in kwargs.keys():
        ax1.fill_between(lags, rho_o2i-kwargs['error_o2i'], rho_o2i+kwargs['error_o2i'],
            alpha=0.2, edgecolor='red', facecolor='red',linewidth=1            
        )
    # Figures details
    ax1.set_ylabel(r"$\rho$", fontsize=15), ax1.set_xlabel(r"$\tau$", fontsize=15)
    ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)
    ax1.set_ylim([0,1]),ax1.set_xlim([lags[0],lags[-1]]), ax1.set_xlabel(r"$\tau$"+"(ms)", fontsize=15)
    # Legend
    plt.legend(fontsize=8, loc='upper right')
    
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