from joblib import Parallel, delayed
import os

# Relative imports
from utils.handle_arguments import initialize_and_grep_files

def run_RCC():    
    from execution.processing_subjects_tools import process_single_subject, process_multiple_subjects
    # Loading the configurations and files with time series 
    opts, files, results_dir, json_config, timeseries_type = initialize_and_grep_files()
    
    if len(files) == 1:
        print("Single subject") 
        print("==============")
        process_single_subject(files[0], opts, results_dir, json_config, format='png')
    else:
        print("Multiple subjects with individual reservoir training") 
        print("====================================================")
        Parallel(n_jobs=opts.num_jobs)(delayed(process_single_subject)(f, opts, results_dir, json_config, format='png') for f in files)

def run_GC():    
    from execution.GC_utils import GC_single_subject
    # Loading the configurations and files with time series 
    opts, files, results_dir, json_config, timeseries_type = initialize_and_grep_files()
    os.remove(os.path.join(results_dir,json_config))

    if len(files) == 1:
        print("Single subject Granger Causality") 
        print("==============")
        GC_single_subject(files[0], opts, results_dir, format='png')
    elif opts.batch_analysis:
        print("Multiple subjects Granger Causality") 
        print("===============================================")
        raise NotImplementedError
    else:
        print("Multiple subjects Granger Causality") 
        print("====================================================")
        Parallel(n_jobs=opts.num_jobs)(delayed(GC_single_subject)(f, opts, results_dir, format='png') for f in files)

if __name__ == '__main__':
    pass