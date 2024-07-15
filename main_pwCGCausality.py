if __name__ == '__main__':
    ## Relative imports
    from methods.estimators.granger_based import pwCGC
    
    # Create arguments to start the process
    for sim_num in [1,7,15,19]: # Simulation 28 should have a minimum lag of -20, otherwise it will raise an error
        for L in [70,75,80,85,90,95,100]:
            args = {
                "dir": f"./Datasets/Netsim/Sim-{sim_num}/Timeseries",
                "r_folder": f"Results_Netsim-Dataset_Sim-{sim_num}_Method-pwCGC/Results_Netsim-Dataset_Sim-{sim_num}_Length-{L}",
                "num_jobs": 4,
                "length": L,
                "subjects": ["-1"],
                "rois": [-1],
                "min_lag": -20, #-10 for simulation 28
                "fmri": True
            }
            
            # Create instance of Bivariate Granger Causality object
            BivariateGrangerCausality = pwCGC(args=args)

            # Fit the whole dataset
            BivariateGrangerCausality.fit_dataset(ic='aic', make_stationary=False)