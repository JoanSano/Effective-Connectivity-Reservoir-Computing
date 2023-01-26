from joblib import Parallel, delayed

# Relative imports
from utils.handle_arguments import initialize_and_grep_files
from execution.processing_subjects_tools import process_single_subject, process_multiple_subjects

def run_RCC():    
    # Loading the configurations and files with time series 
    opts, files, results_dir, json_config, timeseries_type = initialize_and_grep_files()

    
    if len(files) == 1:
        print("Single subject") 
        print("==============")
        process_single_subject(files[0], opts, results_dir, json_config, format='png')
    elif opts.batch_analysis:
        print("Multiple subjects with batch reservoir training") 
        print("===============================================")
        # TODO: Implement k-fold CV
        process_multiple_subjects(files, opts, results_dir, json_config, format='png', name_subject="sub-ignore")
    else:
        print("Multiple subjects with individual reservoir training") 
        print("====================================================")
        Parallel(n_jobs=opts.num_jobs)(delayed(process_single_subject)(f, opts, results_dir, json_config, format='png') for f in files)
    
if __name__ == '__main__':
    pass