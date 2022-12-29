import matplotlib.pylab as plt
import numpy as np

def plot_RCC_input2output(lags, rho_i2o, rho_o2i, **kwargs):
    # Location of the maximum correlation
    max_i2o, max_o2i = lags[np.argmax(rho_o2i)], lags[np.argmax(rho_i2o)]

    # Instantiate figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,15))
    ax2.remove()

    # Plot main curves
    ax1.plot(lags, rho_i2o, linewidth=2, color='blue', label=r'$X \rightarrow Y$' + ' (i.e, ' + r'$x_{t}$' + ' predicts ' + r'$y_{t+\tau}$' + ')')
    ax1.plot(lags, rho_o2i, linewidth=2, color='red', label=r'$Y \rightarrow X$' + ' (i.e, ' + r'$y_{t}$' + ' predicts ' + r'$x_{t+\tau}$' + ')')

    # Plot peaks of correlation
    #ax1.vlines(x=max_o2i, ymin=0, ymax=np.max(rho_i2o), linewidth=1, color='blue', linestyles='--', label=r'$\tau$' + ' = ' + str(max_o2i))
    #ax1.vlines(x=max_i2o, ymin=0, ymax=np.max(rho_o2i), linewidth=1, color='red', linestyles='--', label=r'$\tau$' + ' = ' + str(max_i2o))

    # Legend
    plt.legend(fontsize=13, loc='upper right')

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
    ax1.set_ylim([0 ,1]),ax1.set_xlim([lags[0],lags[-1]]), ax1.set_xticklabels([])

    # Derivative
    left, bottom, width, height = [0.125, 0.325 , 0.775, 0.19]
    der_inset = fig.add_axes([left, bottom, width, height])
    der_inset.spines["top"].set_visible(False), der_inset.spines["right"].set_visible(False)
    der_inset.set_ylabel(r"$\partial_{\tau} \rho$", fontsize=15), der_inset.set_xlabel(r"$\tau$", fontsize=15)
    der_inset.set_xlim([lags[0],lags[-1]])#, der_inset.set_ylim([0 ,1])

    der_inset.plot(lags[:-1], np.diff(rho_i2o), linewidth=2, color='blue')
    der_inset.plot(lags[:-1], np.diff(rho_o2i), linewidth=2, color='red')

    # Visualization
    if 'save' and 'name'in kwargs.keys():
        if 'dpi' in kwargs.keys():
            plt.savefig(kwargs['name'], dpi=kwargs['dpi'])
        else:
            plt.savefig(kwargs['name'])
    else:
        plt.show()

if __name__ == '__main__':
    pass