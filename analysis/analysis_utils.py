import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def unidirectional_score_ij(p_i2j, p_j2i, p_delta_positive, p_delta_negative, lags):
    """
    TODO: Add description. ONeNote for reference.
    """
    Score_x2y = (lags<0)*(1-p_delta_negative)*(1-p_j2i) + (lags>0)*(1-p_delta_positive)*(1-p_i2j)
    Score_y2x = (lags>0)*(1-p_delta_negative)*(1-p_j2i) + (lags<0)*(1-p_delta_positive)*(1-p_i2j)
    return Score_x2y, Score_y2x
    
def score_ij(p_i2j, p_j2i, p_delta):
    """
    TODO: Add description. ONeNote for reference.
    """
    return (1-p_i2j) * (1-p_j2i) * p_delta

def constrain_first_neighbours(GT_net, Pred_net, Mask_N1, scored=False):
    GT_net_flat, Pred_net_flat = [], []
    for i in range(GT_net.shape[0]):
        for j in range(GT_net.shape[1]):
            if Mask_N1[i,j] == 1:
                GT_net_flat.append(np.int16(GT_net[i,j]))               
                if scored:
                    Pred_net_flat.append(Pred_net[i,j])
                else:   
                    Pred_net_flat.append(np.int16(Pred_net[i,j]))
    return GT_net_flat, Pred_net_flat

def constrain_second_neighbours(GT_net, Pred_net, **kwargs):
    # TODO: Implement and/or think about this
    pass

def confusion_matrix_scores(GT_net, Pred_net, **kwargs):
    """
    Computes the sensitivity, the specificity and the positive predictive value of the reconstructed binary networks.
    Inputs:
        GT_net: Ground Truth Network of shape (nodes X nodes) Binary network to which predictions will be comared.
        Pred_net: Reconstruction of the network predicted by the method of shape (nodes X nodes)
        **kwargs
    """

    if "Mask_N1" in kwargs.keys():
        # First neighbours flag
        GT_net_flat, Pred_net_flat = constrain_first_neighbours(GT_net, Pred_net, kwargs["Mask_N1"])
    elif "Mask_N2" in kwargs.keys():
        # Second neighbours flag
        pass
    else:
        GT_net_flat = list(np.int16(GT_net[np.triu_indices_from(GT_net, k=1)]))
        Pred_net_flat = list(np.int16(Pred_net[np.triu_indices_from(Pred_net, k=1)]))
        GT_net_flat.extend(np.int16(GT_net[np.tril_indices_from(GT_net, k=-1)]))
        Pred_net_flat.extend(np.int16(Pred_net[np.tril_indices_from(Pred_net, k=-1)]))

    # Scores & INFO: 
    # -----
    # tn => True negative
    # fp => False positive
    # fn => False negative
    # tp => True positive
    tn, fp, fn, tp = confusion_matrix(GT_net_flat, Pred_net_flat).ravel() 
    sensitivity = tp / (tp + fn) if (tp + fn)>0 else 0
    specificity = tn / (tn + fp) if (tn + fp)>0 else 0
    positive_predictive_value = tp / (tp + fp) if (tp + fp)>0 else 0
    negative_predictive_value = tn / (tn + fn) if (tn + fn)>0 else 0

    return sensitivity, specificity, positive_predictive_value, negative_predictive_value

def roc_analysis(GT_net, Pred_net, **kwargs):
    """
    Performs the Receiver Operating Curve analysis.
    Inputs:
        GT_net: Ground Truth Network of shape (nodes X nodes) Binary network to which predictions will be comared.
        Pred_net: Reconstruction of the network predicted by the method of shape (nodes X nodes)
        **kwargs
    """

    if "Mask_N1" in kwargs.keys():
        # First neighbours flag
        GT_net_flat, Pred_net_flat = constrain_first_neighbours(GT_net, Pred_net, kwargs["Mask_N1"], scored=True)
    elif "Mask_N2" in kwargs.keys():
        # Second neighbours flag
        pass
    else:
        GT_net_flat = list(np.int16(GT_net[np.triu_indices_from(GT_net, k=1)]))
        Pred_net_flat = list(Pred_net[np.triu_indices_from(Pred_net, k=1)])
        GT_net_flat.extend(np.int16(GT_net[np.tril_indices_from(GT_net, k=-1)]))
        Pred_net_flat.extend(Pred_net[np.tril_indices_from(Pred_net, k=-1)])

    # Computing ROC values
    fpr, tpr, thresholds = roc_curve(GT_net_flat, Pred_net_flat)
    auc = roc_auc_score(GT_net_flat, Pred_net_flat)
    
    return fpr, tpr, auc

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
    left, bottom, width, height = [0.12, 0.12 , 0.85, 0.85]
    ax1 = fig.add_axes([left, bottom, width, height])
    # Plot main curves
    for curve in to_plot:    
        ax1.plot(lags, curve["data"], curve["linewidth"], color=curve["color"], linestyle=curve["style"], label=curve["label"], alpha=curve["alpha"])
        ax1.fill_between(lags, curve["data"]-curve["error"], curve["data"]+curve["error"],
            alpha=0.2, edgecolor=curve["color"], facecolor=curve["color"], linewidth=0        
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
    ax1.set_yticks([z_min, 0.5*(z_min+z_max),z_max]), ax1.set_yticklabels([str(z_min), str(0.5*(z_min+z_max)), str(z_max)]), ax1.set_ylim([z_min-0.01,z_max+0.02])
    ax1.set_xlim([lags[0],lags[-1]]), ax1.set_xlabel(kwargs['x_label'], fontsize=15)
    # Legend
    #plt.legend(fontsize=8, frameon=False, ncols=2)
    
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