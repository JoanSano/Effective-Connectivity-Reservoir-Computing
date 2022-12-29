import os
from joblib import Parallel, delayed
import json

## Relative imports
from utils.handle_arguments import handle_argumrnts
from utils.reservoir_networks import return_reservoir_blocks
from subject_utils.single_subject import process_subject

####################
## Execution options
opts, timeseries = handle_argumrnts()

###########################
## Creating necessary paths
root_dir = os.getcwd()
data_dir = opts.dir
results_dir = os.path.join(root_dir, opts.r_folder)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

if __name__ == '__main__':
    #########################################
    ## Initialization of the Reservoir blocks
    I2N, N2N = return_reservoir_blocks(json_file='./reservoir_config.json', exec_args=opts)
    
    ##########################################
    ## Loading the files with time series 
    files = [os.path.join(root_dir, opts.dir, f) for f in os.listdir(opts.dir)]
    process_subject(files[0], opts.rois)
    #Parallel(n_jobs=opts.num_jobs)(delayed(process_subject)(os.path.join(data_dir,f)) for f in files)

