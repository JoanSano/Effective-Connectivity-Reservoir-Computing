import numpy as np
from sklearn.metrics import confusion_matrix

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

def confusion_matrix_scores(GT_net, Pred_net, **kwargs):
    """
    TODO: Add description. ONeNote for reference.
    Only binary networks
    """
    # Flatten the networks to obtain performance

    # TODO: Constrain to 1st neighbours --> normalise with respect to the number of real/direct connections
    if "Mask_N1" in kwargs.keys():
        # First neighbours flag
        Mask_N1 = kwargs["Mask_N1"]
        GT_net_flat, Pred_net_flat = [], []
        for i in range(GT_net.shape[0]):
            for j in range(GT_net.shape[1]):
                if Mask_N1[i,j] == 1:
                    GT_net_flat.append(np.int16(GT_net[i,j]))               
                    Pred_net_flat.append(np.int16(Pred_net[i,j]))
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

    return sensitivity, specificity, positive_predictive_value