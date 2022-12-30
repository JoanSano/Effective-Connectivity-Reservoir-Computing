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
    max_i2o, max_o2i = lags[np.argmax(rho_o2i)], lags[np.argmax(rho_i2o)]

    # Instantiate figure
    fig, ax = plt.subplots(figsize=(10,8))
    ax.remove()

    ##################
    left, bottom, width, height = [0.1, 0.325 , 0.85, 0.65]
    ax1 = fig.add_axes([left, bottom, width, height])
    # Plot main curves
    ax1.plot(lags, rho_i2o, linewidth=2, color='blue', label=X+r'$_{t}$'+' predicts '+Y+r'$_{t+\tau}$')
    ax1.plot(lags, rho_o2i, linewidth=2, color='red', label=Y+r'$_{t}$'+' predicts '+X+r'$_{t+\tau}$')
    # Plot peaks of correlation
    #ax1.vlines(x=max_o2i, ymin=0, ymax=np.max(rho_i2o), linewidth=1, color='blue', linestyles='--', label=r'$\tau$' + ' = ' + str(max_o2i))
    #ax1.vlines(x=max_i2o, ymin=0, ymax=np.max(rho_o2i), linewidth=1, color='red', linestyles='--', label=r'$\tau$' + ' = ' + str(max_i2o))
    # Display statistics (if present)
    if 'error_i2o' in kwargs.keys():
        ax1.fill_between(lags, rho_i2o-kwargs['error_i2o'], rho_i2o+kwargs['error_i2o'],
            alpha=0.2, edgecolor='blue', facecolor='blue',linewidth=1            
        )
    if 'error_o2i' in kwargs.keys():
        ax1.fill_between(lags, rho_o2i-kwargs['error_o2i'], rho_o2i+kwargs['error_o2i'],
            alpha=0.2, edgecolor='red', facecolor='red',linewidth=1            
        )
    # Figures details
    ax1.set_ylabel(r"$\rho$", fontsize=15)#, ax1.set_xlabel(r"$\tau$", fontsize=15)
    ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)
    ax1.set_ylim([-0.3,1]),ax1.set_xlim([lags[0],lags[-1]]), ax1.set_xticklabels([])
    # Legend
    plt.legend(fontsize=13, loc='upper right')

    ############
    left, bottom, width, height = [0.1, 0.08 , 0.85, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    # Derivative
    ax2.plot(lags[:-1], np.diff(rho_i2o), linewidth=2, color='blue')
    ax2.plot(lags[:-1], np.diff(rho_o2i), linewidth=2, color='red')
    # Figures details
    ax2.spines["top"].set_visible(False), ax2.spines["right"].set_visible(False)
    ax2.set_ylabel(r"$\partial_{\tau} \rho$", fontsize=15), ax2.set_xlabel(r"$\tau$", fontsize=15)
    ax2.set_xlim([lags[0],lags[-1]])

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

if __name__ == '__main__':
    pass