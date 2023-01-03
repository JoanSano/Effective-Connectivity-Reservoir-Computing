import numpy as np
import os

# Relative imports
from utils.handle_arguments import handle_argumrnts

def initialize_and_grep_files():
    """
    TODO: Add documentation
    """
    
    # Execution options
    opts, timeseries = handle_argumrnts()
    
    # Creating necessary paths
    root_dir = os.getcwd()
    results_dir = os.path.join(root_dir, opts.r_folder)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        
    # Drop command line call for transparency
    with open(os.path.join(results_dir, 'commandline_args.txt'), 'w') as f:
        for arg, val in zip(opts.__dict__.keys(),opts.__dict__.values()):
            f.write(arg+': '+str(val)+'\n')

    # Reservoir Architecture parameters file
    json_config = './reservoir_config.json'
    os.system(f"cp {json_config} {results_dir}")

    # Corresponding data files
    files = [os.path.join(opts.dir, f) for f in os.listdir(opts.dir) if f.split(".")[0] in opts.subjects or opts.subjects[0] == '-1']

    return opts, files, results_dir, json_config

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

    if split == 100:
        # No split. Train and test data are the same
        x_train = input
        y_train = output

        x_test = input
        y_test = output
    elif split == -1: 
        # 1-fold cross validation
        x_train = input[:-1,...]
        y_train = output[:-1,...]
        # shape is kept as (1,T) for further compatibility
        x_test = input[-1,...].reshape(1,-1)
        y_test = output[-1,...].reshape(1,-1)
    # TODO: implement k-fold cross validation
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
