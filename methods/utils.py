import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from utils.training_utils import input_output_lagged, split_train_test_reshape, prepare_data
from methods.reservoir_networks import reservoir_network

def reservoir_input2output(input, output, lag, I2N, N2N, split=75, skip=20, shuffle=False, axis=0, runs=None):
    """
    TODO: Add description
    """
    # Rolling the time series
    input_data, output_data = input_output_lagged(input, output, lag, axis=axis)
    
    # Split data in train and test
    input_train, output_train, input_test, output_test = split_train_test_reshape(input_data, output_data, split, shuffle=shuffle, axis=axis)
    
    # Prepare data
    input_train, output_train, input_test, output_test = prepare_data(input_train, output_train, input_test, output_test)
    
    
    # Fit and predict output(t+t*) from input(t)
    if not runs:
        reservoir_i2o = reservoir_network(I2N, N2N)
        reservoir_i2o.fit(X=input_train, y=output_train)
        output_pred = reservoir_i2o.predict(input_test)
    else:
        output_pred = np.empty(shape=(runs,), dtype=object)
        test_data_copy = np.empty(shape=(runs,), dtype=object)
        for run in range(runs):
            reservoir_i2o = reservoir_network(I2N, N2N)
            reservoir_i2o.fit(X=input_train, y=output_train)
            output_pred[run] = reservoir_i2o.predict(input_test)[0]
            test_data_copy[run] = output_test[0]
        output_test = test_data_copy
    
    # Predictability measured by the correlation between ground-truth and prediction
    correlations = np.zeros((output_test.shape[0],))
    ground_truth, predictions = np.zeros((output_test.shape[0], output_test[0].shape[0]-skip)), np.zeros((output_test.shape[0], output_test[0].shape[0]-skip))
    
    for i, (x, y) in enumerate(zip(output_test, output_pred)):
        correlations[i] = np.corrcoef(x[skip:],y[skip:])[0,1]
        ground_truth[i] = x[skip:]
        predictions[i] = y[skip:]

    return correlations, ground_truth, predictions

def RCC(input, output, lags, I2N, N2N, split=75, skip=20, shuffle=False, axis=0, runs=None):
    """
    TODO: Add description
    """

    # Try different time lags
    results_i2o = pd.DataFrame(columns=["lag","predictability","ground_truth","predictions"])
    results_o2i = pd.DataFrame(columns=["lag","predictability","ground_truth","predictions"])
    for lag in lags:
        # x(t) predicts y(t+t*)
        correlations, ground_truth, predictions = reservoir_input2output(input, output, lag, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis, runs=runs)
        results_i2o.loc[len(results_i2o.index)] = [lag, correlations, ground_truth, predictions]
        
        # y(t) predicts x(t+t*)
        correlations, ground_truth, predictions = reservoir_input2output(output, input, lag, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis, runs=runs)
        results_o2i.loc[len(results_o2i.index)] = [lag, correlations, ground_truth, predictions]

    return results_i2o, results_o2i

def RCC_average(x, y, lags, I2N, N2N, split=75, skip=20, shuffle=False, axis=0, runs=None, average=False):
    """
    TODO: Add description
    """

    # Reservoir Computing Causality - which needs to be tested accross several lags
    results_x2y, results_y2x = RCC(x, y, lags, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis, runs=runs)

    # We extract the data
    if not runs:
        Nsamples = results_x2y["predictability"][0].shape[0] 
    else:
        Nsamples = runs
    corr_x2y, corr_y2x, sem_x2y, sem_y2x = np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples))
    for i in range(lags.shape[0]):
        corr_x2y[i] = results_x2y["predictability"][i]
        corr_y2x[i] = results_y2x["predictability"][i]
    
    # Stats 
    if average:
        mean_x2y, mean_y2x = np.mean(corr_x2y, axis=1), np.mean(corr_y2x, axis=1)
        sem_x2y, sem_y2x = np.std(corr_x2y, axis=1)/np.sqrt(Nsamples), np.std(corr_y2x, axis=1)/np.sqrt(Nsamples)
        
        return np.expand_dims(mean_x2y, axis=1), np.expand_dims(mean_y2x, axis=1), results_x2y.drop("predictability", axis=1), results_y2x.drop("predictability", axis=1)
    else:
        return corr_x2y, corr_y2x, results_x2y.drop("predictability", axis=1), results_y2x.drop("predictability", axis=1)

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

def directionality_test(x2y, y2x, surrogate_x2y, surrogate_y2x, lags, significance=0.05, permutations=False, axis=1, bonferroni=True):
    """
    TODO: Add description. ONeNote for reference.
    """
    
    # Delta: Difference in predictability (McCracken & Weigel Phys. Rev. E. 2014)
    Delta, Delta_surrogate = x2y - y2x, surrogate_x2y - surrogate_y2x
    _, p_delta_positive = ttest_ind(Delta, Delta_surrogate, axis=axis, equal_var=False, permutations=permutations, alternative='greater')
    _, p_delta_negative = ttest_ind(Delta, Delta_surrogate, axis=axis, equal_var=False, permutations=permutations, alternative='less')
    _, p_delta = ttest_ind(Delta, Delta_surrogate, axis=axis, equal_var=False, permutations=permutations, alternative='two-sided')
    
    # Predictabilities are statistically significant
    _, p_x2y = ttest_ind(x2y, surrogate_x2y, axis=axis, equal_var=False, permutations=permutations, alternative='greater')
    _, p_y2x = ttest_ind(y2x, surrogate_y2x, axis=axis, equal_var=False, permutations=permutations, alternative='greater')
    
    # Causality Scores    
    Score_x2y, Score_y2x = unidirectional_score_ij(p_x2y, p_y2x, p_delta_positive, p_delta_negative, lags)
    Score_xy = score_ij(p_x2y, p_y2x, p_delta)
    
    # Statistical evidence: Compute the scores at the critical values (one-sided and two sided respectively)
    if bonferroni:
        # Num hypothesis is 3
        threshold_uni, _ = unidirectional_score_ij(significance/2, significance/2, significance/2, significance/2, -1)
        threshold_bi = score_ij(significance/3, significance/3, significance/(2*3))
    else:
        threshold_uni, _ = unidirectional_score_ij(significance, significance, significance, significance, -1)
        threshold_bi = score_ij(significance, significance, significance/2)
    
    evidence_x2y = np.where(Score_x2y>=threshold_uni, 1, np.nan)
    evidence_y2x = np.where(Score_y2x>=threshold_uni, 1, np.nan)
    evidence_xy = np.where(Score_xy>=threshold_bi, 1, np.nan)

    return evidence_xy, evidence_x2y, evidence_y2x, Score_xy, Score_x2y, Score_y2x

if __name__ == '__main__':
    pass