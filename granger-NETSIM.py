if __name__ == '__main__':
    from methods.estimators.granger_based import pwGC, pwCGC, Spectral_pwCGC

    method = "spwCGC" # "pwCGC" # "pwGC"

    # Create arguments to start the process
    for sim_num in [28]: #1,7,15,19, 28
        for L in [100]:#70,75,80,85,90,95,
            args = {
                "dir": f"./Datasets/Netsim/Sim-{sim_num}/Timeseries",
                "r_folder": f"Results_Netsim-Dataset_Sim-{sim_num}_Method-{method}/Results_Netsim-Dataset_Sim-{sim_num}_Length-{L}",
                "num_jobs": 4,
                "length": L,
                "subjects": ["-1"],
                "rois": [-1],
                "min_lag": -1,
                "num_surrogates":500,
                "fmri": True
            }   
            
            if method == "pwGC":                
                pwGC(args=args).fit_dataset(run_self_loops=False, make_stationary=False, plot=False)

            elif method == "pwCGC":                
                pwCGC(args=args).fit_dataset(ic='aic', make_stationary=False, plot=False)

            elif method == "spwCGC":
                Spectral_pwCGC(args=args).fit_dataset(ic='aic', make_stationary=True, save_surrogates=True, plot=False)

            else:
                raise ValueError("Granger method not implemented!")

