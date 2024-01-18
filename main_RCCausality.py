if __name__ == '__main__':
    from joblib import Parallel, delayed

    ## Relative imports
    from utils.handle_arguments import initialize_and_grep_files  
    from methods.effective_connectivity import RCC
    
    # Loading the configurations and files with time series 
    opts, files, results_dir, json_config, timeseries_type = initialize_and_grep_files()
    ReservoirComputingCausality = RCC(results_dir, opts, json_config)
    
    if len(files) == 1:
        print("Single subject") 
        print("==============")
        ReservoirComputingCausality.fit(files[0], format='png', factor=10, verbose=True)
    else:
        print("Multiple subjects with individual reservoir training") 
        print("====================================================")
        Parallel(n_jobs=opts.num_jobs)(delayed(ReservoirComputingCausality.fit)(
            f, opts, results_dir, json_config, format='png', factor=10, verbose=True) for f in files
        )