import numpy as np
import matplotlib.pylab as plt

def input_output_lagged(input,output,target_lag):
    target_lag = int(target_lag)
    if target_lag > 0:
        # Here we manually roll the target time series so the Reservoir predicts the time series at a given time lag
        # x(t) --> x(t-t*) where t* is the lag
        x_data = input[:-target_lag]
        y_data = output[target_lag:]
        return x_data, y_data
    elif target_lag < 0:
        # Here we manually roll the target time series so the Reservoir predicts the time series at a given time lag
        # x(t) --> x(t+t*) where t* is the lag
        x_data = input[-target_lag:]
        y_data = output[:target_lag]
        return x_data, y_data
    else: 
        # If no lag, then return the data as it is
        x_data = input
        y_data = output
        return x_data, y_data

def split_train_test_reshape(input,output,split):
    Remaining_points = output.shape[0]
    assert Remaining_points == input.shape[0]

    # Split data in train and test
    limit = int(Remaining_points*0.01*split)
    x_train = input[:limit].reshape(-1, 1)
    y_train = output[:limit]

    x_test = input[limit:].reshape(-1, 1)
    y_test = output[limit:]

    return x_train, y_train, x_test, y_test

<<<<<<< HEAD
<<<<<<< HEAD
if __name__ == '__main__':
    pass
=======
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510
=======
>>>>>>> 1cf6832f6b18363625f35930ef44e76dc778b510
