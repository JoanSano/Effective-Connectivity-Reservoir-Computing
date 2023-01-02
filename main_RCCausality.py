import os
from joblib import Parallel, delayed

## Relative imports
from utils.handle_arguments import handle_argumrnts
from subject_utils.processing_subjects_tools import process_subject

####################
## Execution options
opts, timeseries = handle_argumrnts()

###########################
## Creating necessary paths
root_dir = os.getcwd()
results_dir = os.path.join(root_dir, opts.r_folder)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    
# Drop command line call for transparency
with open(os.path.join(results_dir, 'commandline_args.txt'), 'w') as f:
    for arg, val in zip(opts.__dict__.keys(),opts.__dict__.values()):
        f.write(arg+': '+str(val)+'\n')

if __name__ == '__main__':
    ########################################
    # Reservoir Architecture parameters file
    json_config = './reservoir_config.json'

    ##########################################
    ## Loading the files with time series 
    # TODO: Test for a specific selection of subjects from command line
    files = [os.path.join(opts.dir, f) for f in os.listdir(opts.dir)]
    process_subject(files[0], opts, results_dir, json_config, format='png')

    # TODO: Parallelize subjects
    #Parallel(n_jobs=opts.num_jobs)(delayed(process_subject)(os.path.join(data_dir,f)) for f in files)

