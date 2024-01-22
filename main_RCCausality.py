if __name__ == '__main__':
    from joblib import Parallel, delayed

    ## Relative imports
    from utils.handle_arguments import initialize_and_grep_files  
    from methods.effective_connectivity import RCC
    
    # Create instance of RCC
    ReservoirComputingCausality = RCC()

    # Get execution configs
    files = ReservoirComputingCausality.files
    num_jobs = ReservoirComputingCausality.opts.num_jobs
    
    if len(files) == 1:
        print("Single subject") 
        print("==============")
        ReservoirComputingCausality.fit_subject(files[0], format='png', factor=10, verbose=True)
    else:
        print("Multiple subjects with individual reservoir training") 
        print("============== Parallel processing ==================")
        Parallel(n_jobs=num_jobs)(delayed(ReservoirComputingCausality.fit_subject)(
            f, format='png', factor=10, verbose=False) for f in files
        )