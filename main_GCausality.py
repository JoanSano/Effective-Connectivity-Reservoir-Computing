if __name__ == '__main__':
    from joblib import Parallel, delayed
    import os

    ## Relative imports
    from utils.handle_arguments import initialize_and_grep_files
    from GrangerCausality.compute_effective_connectivity import process_single_subject

    # Loading the configurations and files with time series 
    opts, files, results_dir, json_config, timeseries_type = initialize_and_grep_files()
    os.remove(os.path.join(results_dir,json_config))

    if len(files) == 1:
        print("Single subject Granger Causality") 
        print("==============")
        process_single_subject(files[0], opts, results_dir, format='png')
    else:
        print("Multiple subjects Granger Causality") 
        print("====================================================")
        Parallel(n_jobs=opts.num_jobs)(delayed(process_single_subject)(f, opts, results_dir, format='png') for f in files)