import numpy as np

from utils.training_utils import input_output_lagged, split_train_test_reshape
from utils.reservoir_networks import reservoir_network

def reservoir_input2output(input, output, lag, I2N, N2N, split=75, skip=20):
    # Rolling the time series
    input_data, output_data = input_output_lagged(input, output, lag)
    Remaining_points = output_data.shape[0]

    # Split data in train and test
    input_train, output_train, input_test, output_test = split_train_test_reshape(input_data, output_data, split)

    # Fit and predict output(t+t*) from input(t)
    reservoir_i2o = reservoir_network(I2N, N2N)
    reservoir_i2o.fit(X=input_train, y=output_train)
    output_pred = reservoir_i2o.predict(input_test)
    
    return np.corrcoef(output_test[skip:],output_pred[skip:])[0,1], [output_test, output_pred]

def RCC_input2output(input, output, lags, I2N, N2N, split=75, skip=20):
    # Try different time lags
    correlations_i2o, correlations_o2i = [], []
    results_i2o, results_o2i = [], []
    for lag in lags:
        # x(t) predicts y(t+t*)
        rho, results = reservoir_input2output(input, output, lag, I2N, N2N, split=split, skip=skip)
        results_i2o.append(results)
        correlations_i2o.append(rho)

        # y(t) predicts x(t+t*)
        rho, results = reservoir_input2output(output, input, lag, I2N, N2N, split=split, skip=skip)
        results_o2i.append(results)
        correlations_o2i.append(rho)
    return correlations_i2o, correlations_o2i, results_i2o, results_o2i

def RCC_statistics(x, y, lags, runs, I2N, N2N, split=75, skip=20, return_results=True):
    # We run several reservoirs in parallel
    if return_results:
        correlations_x2y, correlations_y2x, results_x2y, results_y2x = np.zeros((runs, lags.shape[0])), np.zeros((runs, lags.shape[0])), [], []

    if return_results:
        for run in range(runs):
            # Single RCC run
            correlations_x2y[run,:], correlations_y2x[run,:], r_x2y, r_y2x = RCC_input2output(x, y, lags, I2N, N2N, split=split, skip=skip)
            results_x2y.append(r_x2y)
            results_y2x.append(r_y2x)
        return correlations_x2y, correlations_y2x, results_x2y, results_y2x
    else:
        for run in range(runs):
            # Single RCC run
            RCC_input2output(x, y, lags, I2N, N2N, split=split, skip=skip)