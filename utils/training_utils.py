import numpy as np

def input_output_lagged(input, output, target_lag, axis=0):
    """
    TODO: Add description
    initial dims: subjects X time-points
    """

    if axis != 0:
        np.swapaxes(input, 0, axis), np.swapaxes(output, 0, axis) 

    # Roll the target time series so the Reservoir predicts the time series at a given time lag
    target_lag = int(target_lag)
    if target_lag > 0:
        # x(t) --> x(t-t*) where t* is the lag
        x_data = input[:,:-target_lag]
        y_data = output[:,target_lag:]
    elif target_lag < 0:
        # x(t) --> x(t+t*) where t* is the lag
        x_data = input[:,-target_lag:]
        y_data = output[:,:target_lag]
    else: 
        # If no lag, then return the data as it is
        x_data = input
        y_data = output
    
    if axis != 0:
        np.swapaxes(x_data, 0, axis), np.swapaxes(y_data, 0, axis) 
    return x_data, y_data

def split_train_test_reshape(input, output, split, shuffle=False, axis=0):
    """
    TODO: Add description
    The split is done according to the first dimension of the array or accross the items of the list.
    """
    
    # Dimensions of arrays need to be equal (input --> target)
    assert output.shape == input.shape
    
    if axis != 0:
        input, output = np.swapaxes(input, 0, axis), np.swapaxes(output, 0, axis) 
        
    # Shuffle data (in development)
    if shuffle:
        # TODO: Check is it keeps the same input-output relationships!
        indices = np.arange(input.shape[0])
        np.random.shuffle(indices)
        input, output = input[indices], output[indices]

    if split == 100 or split == 0:
        # No split. Train and test data are the same
        x_train = input
        y_train = output

        x_test = input
        y_test = output
    elif split <= -1: # k-fold cross validation
        x_train = input[:split,...]
        y_train = output[:split,...]
        
        x_test = input[split:,...]
        y_test = output[split:,...]
    else:
        # Split data in train and test
        limit = int(output.shape[0]*0.01*split)
        x_train = input[:limit,...]
        y_train = output[:limit,...]

        x_test = input[limit:,...]
        y_test = output[limit:,...]
        
    if axis != 0:
        x_train, y_train = np.swapaxes(x_train, 0, axis), np.swapaxes(y_train, 0, axis) 
        x_test, y_test = np.swapaxes(x_test, 0, axis), np.swapaxes(y_test, 0, axis) 
    
    # Adding an extra dimension to training data --> n_features=1
    return  np.expand_dims(x_train, axis=-1), y_train,  np.expand_dims(x_test, axis=-1),  y_test

def prepare_data(*args, axis=0):
    """
    TODO: Add documentation
    """

    # For each argument we prepare an empty numpy array
    prepared_data = []
    for j, array in enumerate(args):
        n_sequences = args[j].shape[axis] 
        temp_data = np.empty(shape=n_sequences, dtype=object)
        for i in range(n_sequences):
            temp_data[i] = array[i,...]
        prepared_data.append(temp_data)
    return prepared_data

if __name__ == '__main__':
    pass
