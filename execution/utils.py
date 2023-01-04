
# Relative imports
from utils.handle_arguments import initialize_and_grep_files
from execution.processing_subjects_tools import process_single_subject, process_multiple_subjects

def run_RCC():    
    # Loading the configurations and files with time series 
    opts, files, results_dir, json_config, timeseries_type = initialize_and_grep_files()
    
    if not opts.batch_analysis or len(files) == 1:
        print("Single subject")
        process_single_subject(files[0], opts, results_dir, json_config, format='png')
    else:
        print("Multiple subjects")
        process_multiple_subjects(files, opts, results_dir, json_config, format='png')

# TODO: Where, if possible, can I parallelize?
#Parallel(n_jobs=opts.num_jobs)(delayed(process_subject)(os.path.join(data_dir,f)) for f in files)

if __name__ == '__main__':
    pass