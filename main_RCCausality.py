from joblib import Parallel, delayed

## Relative imports
from utils.handle_arguments import initialize_and_grep_files
from subject_utils.processing_subjects_tools import process_single_subject, process_multiple_subjects

if __name__ == '__main__':
    ########################################
    # Loading the configurations and files with time series 
    opts, files, results_dir, json_config, timeseries_type = initialize_and_grep_files()
    
    # TODO: Add flag for batch or single processing 
    # NOTE: When a single subject is used, it should identify that flag as single subject

    process_single_subject(files[0], opts, results_dir, json_config, format='png')
    
    # TODO: Where, if possible, can I parallelize?
    #Parallel(n_jobs=opts.num_jobs)(delayed(process_subject)(os.path.join(data_dir,f)) for f in files)

    #process_multiple_subjects(files, opts, results_dir, json_config, format='png')