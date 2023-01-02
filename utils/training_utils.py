import numpy as np
import matplotlib.pylab as plt

def input_output_lagged(input, output, target_lag):
    """
    TODO: Add description
    """

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

def split_train_test_reshape(input, output, split, axis=0):
    """
    TODO: Add description
    """

    # Swap the axis you want the split to be applied and swap with the 0-th
    # Should only be used for 2D arrays although the code will work for higher dimensional cases
    swapped_input = np.swapaxes(input, 0, axis)
    swapped_output = np.swapaxes(output, 0, axis)
    assert swapped_output.shape[0] == swapped_input.shape[0]

    if split == 100:
        # No split. Train and test data are the same
        x_train = swapped_input
        y_train = swapped_output

        x_test = swapped_input
        y_test = swapped_output
    elif split == -1: 
        # 1-fold cross validation
        x_train = swapped_input[:-1,...]
        y_train = swapped_output[:-1,...]

        x_test = swapped_input[-1,...]
        y_test = swapped_output[-1,...]
    # TODO: implement k-fold cross validation
    else:
        # Split data in train and test
        limit = int(swapped_output.shape[0]*0.01*split)
        x_train = swapped_input[:limit,...]
        y_train = swapped_output[:limit,...]

        x_test = swapped_input[limit:,...]
        y_test = swapped_output[limit:,...]
   
    if len(swapped_input.shape) == 1:
        return x_train.reshape(-1,1), y_train, x_test.reshape(-1,1), y_test
    elif split == -1:
        return x_train.T, y_train.T, x_test.reshape(-1,1), y_test.T
    else:  
        # Re-swap the axis so that the split is positioned in the 1st axis (or the second dimension)
        return  x_train.T, y_train.T,  x_test.T,  y_test.T

if __name__ == '__main__':
    """
    Program description:
    --------------------
    Code snipped to test that the spliting of the data correct. 
    You can use this code to make sure that the input and target 
        are splitted along the desired axis. It returns all the 
        array shapes that are going to be fitted to the reservoir.
    Obviously, it reshapes everything according to PyRCN specs.
    """

    import sys
    axis = int(sys.argv[1])
    x = np.random.rand(500)
    print("Array shape: ", x.shape)
    print("Axis to split: ", axis)
    a, b, c, d = split_train_test_reshape(x,x,split=-1, axis=axis)
    print("Train and test input shapes: ", a.shape, c.shape)
    print("Train and test target shapes: ", b.shape, d.shape)
