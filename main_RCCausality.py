if __name__ == '__main__':
    ## Relative imports
    from methods.effective_connectivity import RCC
    
    # Create instance of RCC
    ReservoirComputingCausality = RCC()

    # Fit the whole dataset -- number of jobs is determined by the user
    ReservoirComputingCausality.fit_dataset(run_self_loops=False, factor=10)