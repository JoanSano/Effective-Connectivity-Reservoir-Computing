if __name__ == '__main__':
    from joblib import Parallel, delayed

    ## Relative imports
    from utils.handle_arguments import initialize_and_grep_files  
    from ReservoirComputingCausality.compute_effective_connectivity import process_single_subject
    
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