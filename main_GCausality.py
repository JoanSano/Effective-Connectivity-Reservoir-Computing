if __name__ == '__main__':
    ## Relative imports
    from methods.compute_ec import bivariate_GC
    
    # Create instance of Bivariate Granger Causality object
    BivariateGrangerCausality = bivariate_GC()

    # Fit the whole dataset
    BivariateGrangerCausality.fit_dataset(run_self_loops=False, make_stationary=False)