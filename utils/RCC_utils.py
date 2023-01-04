import numpy as np
import pandas as pd

from utils.training_utils import input_output_lagged, split_train_test_reshape, prepare_data
from utils.reservoir_networks import reservoir_network

def reservoir_input2output(input, output, lag, I2N, N2N, split=75, skip=20, shuffle=False, axis=0):
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
    reservoir_i2o = reservoir_network(I2N, N2N)
    reservoir_i2o.fit(X=input_train, y=output_train)
    output_pred = reservoir_i2o.predict(input_test)

    # Predictability measured by the correlation between ground-truth and prediction
    correlations = np.zeros((output_test.shape[0],))
    ground_truth, predictions = np.zeros((output_test.shape[0], output_test[0].shape[0]-skip)), np.zeros((output_test.shape[0], output_test[0].shape[0]-skip))
    
    for i, (x, y) in enumerate(zip(output_test, output_pred)):
        correlations[i] = np.corrcoef(x[skip:],y[skip:])[0,1]
        ground_truth[i] = x[skip:]
        predictions[i] = y[skip:]

    return correlations, ground_truth, predictions

def RCC(input, output, lags, I2N, N2N, split=75, skip=20, shuffle=False, axis=0):
    """
    TODO: Add description
    """

    # Try different time lags
    results_i2o = pd.DataFrame(columns=["lag","predictability","ground_truth","predictions"])
    results_o2i = pd.DataFrame(columns=["lag","predictability","ground_truth","predictions"])
    for lag in lags:
        # x(t) predicts y(t+t*)
        correlations, ground_truth, predictions = reservoir_input2output(input, output, lag, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis)
        results_i2o.loc[len(results_i2o.index)] = [lag, correlations, ground_truth, predictions]
        
        # y(t) predicts x(t+t*)
        correlations, ground_truth, predictions = reservoir_input2output(output, input, lag, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis)
        results_o2i.loc[len(results_o2i.index)] = [lag, correlations, ground_truth, predictions]

    return results_i2o, results_o2i

def RCC_statistics(x, y, lags, I2N, N2N, split=75, skip=20, shuffle=False, axis=0):
    """
    TODO: Add description
    """

    # Reservoir Computing Causality - which needs to be tested accross several lags
    results_x2y, results_y2x = RCC(x, y, lags, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis)

    # We extract the data
    Nsamples = x.shape[0]    
    corr_x2y, corr_y2x, sem_x2y, sem_y2x = np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples))
    for i in range(lags.shape[0]):
        corr_x2y[i] = results_x2y["predictability"][i]
        corr_y2x[i] = results_y2x["predictability"][i]

    # Stats 
    mean_x2y, sem_x2y = np.mean(corr_x2y, axis=1), np.std(corr_x2y, axis=1)/np.sqrt(Nsamples)
    mean_y2x, sem_y2x = np.mean(corr_y2x, axis=1), np.std(corr_y2x, axis=1)/np.sqrt(Nsamples)

    return mean_x2y, sem_x2y, mean_y2x, sem_y2x, results_x2y, results_y2x
            
if __name__ == '__main__':
    pass