import numpy as np
from joblib import Parallel, delayed
import os
import warnings
from statsmodels.tsa.stattools import adfuller

## Relative imports
from methods.utils import RCC_average, directionality_test_RCC, directionality_test_GC
from methods.reservoir_networks import return_reservoir_blocks
from utils.surrogates.surrogate_tools import create_surrogates, surrogate_reservoirs
from analysis.utils import generate_report
from utils.handle_arguments import initialize_and_grep_files  

class RCC():
    def __init__(self, args=None) -> None:
        # Loading the configurations and files with time series 
        self.opts, self.files, self.output_dir, self.json_file_config, self.timeseries_type = initialize_and_grep_files(args=args)
        
        # Initialization of the Reservoir blocks
        self.I2N, self.N2N = return_reservoir_blocks(
            json_file=self.json_file_config, exec_args=self.opts
        )

        # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
        self.lags = np.arange(self.opts.min_lag, self.opts.max_lag)
        
        # Load RCC config 
        self.length, self.split = self.opts.length, self.opts.split,
        self.skip, self.runs, self.N_surrogates =  self.opts.skip, self.opts.runs, self.opts.num_surrogates

    def fit_subject(
            self, subject_file, run_self_loops=False, factor=10, verbose=True
        ):
        """
        TODO: Add description of the function

        Arguments
        -----------
        subject_file: (string) Full path to the file containing the time series. ROI time series are stored as columns.
        TODO: finish arguments

        Outputs
        -----------
        TODO: Add output description.
        """
        
        name_subject = subject_file.split("/")[-1].split("_TS")[0] + '_Length-' + str(self.length) + '_Method-RCC'
        print(f"Participant ID: {name_subject}")
        if verbose:
            print("Loading data")

        # Load time series from subject -- dims: time-points X total-ROIs
        time_series = np.genfromtxt(subject_file, delimiter='\t')
        if np.isnan(time_series[:,0]).all():
            time_series = time_series[:,1:] # First column is dropped due to Nan
        limit = int(time_series.shape[0]*0.01*self.length)

        # ROIs from input command
        self.ROIs = list(range(time_series.shape[-1])) if self.opts.rois[0] == -1 else [roi-1 for roi in self.opts.rois]
        self.ROIs = sorted(self.ROIs)

        # Time series to analyse -- dims: ROIs X 1 X time-points
        TS2analyse = np.expand_dims(
            np.array([time_series[:limit,roi] for roi in self.ROIs]), axis=1
        )
        
        # Generate and predict surrogates
        if len(self.ROIs)<=(factor/2 + 1):
            surrogate_population = create_surrogates(TS2analyse, self.ROIs, self.N_surrogates, factor=factor)
        else:
            surrogate_population = None
        if verbose:
            print("Done!")
            print("-----")
            print("Computing reservoir scores and evidence")

        # Compute RCC causality
        for i, roi_i in enumerate(self.ROIs):
            for j in range(i if run_self_loops else i+1, len(self.ROIs)):
                roi_j = self.ROIs[j]
                
                # Run RCC by splitting on axis #1 (i.e., the time points)
                if verbose:
                    print(f"Training reservoirs for ROIs [{roi_i},{roi_j}]")
                x2y, y2x, _, _ = RCC_average(
                    TS2analyse[i], TS2analyse[j], 
                    self.lags, self.I2N, self.N2N, split=self.split, skip=self.skip, 
                    shuffle=False, axis=1, runs=self.runs, average=False
                )
                if verbose:
                    print("Done!")
                    print("-----")
                
                # IAAFT surrogates test
                    print(f"Training surrogate reservoirs for ROIs [{roi_i},{roi_j}]")
                surrogate_x2y, surrogate_y2x = surrogate_reservoirs(
                    TS2analyse[i], TS2analyse[j], self.N_surrogates, 
                    self.lags, self.I2N, self.N2N, self.split, self.skip, 
                    surrogate_population, verbose=verbose
                )  
                if verbose:
                    print("Done!")
                    print("-----")

                # RCC Scores
                if verbose:
                    print(f"Estimating the directionality for ROIs [{roi_i},{roi_j}]")
                evidence_xy, evidence_x2y, evidence_y2x, Score_xy, Score_x2y, Score_y2x = directionality_test_RCC(
                    x2y, y2x, surrogate_x2y, surrogate_y2x, self.lags, 
                    significance=0.05, permutations=False, axis=1, bonferroni=True
                )
                if verbose:
                    print("Done!")
                    print("-----")

                # Generate report
                    print(f"Saving the summary for ROIs [{roi_i},{roi_j}]")
                generate_report(
                    self.output_dir, name_subject, roi_i, roi_j,
                    self.lags, x2y, y2x, surrogate_x2y, surrogate_y2x,
                    Score_x2y, Score_y2x, Score_xy, 
                    evidence_x2y, evidence_y2x, evidence_xy
                )
                if verbose:
                    print("Done!")
                    print("-----")
        
        print("Subject finished!")
        print("-------------------------------")
        return name_subject
    
    def fit_dataset(
            self, run_self_loops=False, factor=10
        ):
        """
        TODO: Add description of the function

        Arguments
        -----------
        subject_file: (string) Full path to the file containing the time series. ROI time series are stored as columns.
        TODO: finish arguments

        Outputs
        -----------
        TODO: Add output description.
        """

        name_subjects = []
        if self.opts.num_jobs == 1:
            print("Multiple subjects with individual reservoir training") 
            print("============= Sequential processing =================")
            print("INFO: \t Parallel or sequential processing depends on the input arguments --num_jobs")
            for f in self.files:
                name_subjects.append(
                    self.fit_subject(f, run_self_loops=run_self_loops, factor=factor, verbose=False)
                )
        else:
            print("Multiple subjects with individual reservoir training") 
            print("============== Parallel processing ==================")
            name_subjects = Parallel(n_jobs=self.opts.num_jobs)(
                delayed(self.fit_subject)(f, run_self_loops=run_self_loops, factor=factor, verbose=False)
                for f in self.files
            )

class bivariate_GC():
    def __init__(self, args=None) -> None:
        # Loading the configurations and files with time series 
        self.opts, self.files, self.output_dir, json_file_config, self.timeseries_type = initialize_and_grep_files(args=args)
        os.remove(os.path.join(self.output_dir,json_file_config)) # No reservoir needed

       # Lags to test; in this scenario, always negative
        min_lag = np.abs(self.opts.min_lag)
        self.lags = np.arange(1,min_lag+1)

        # Load config 
        self.length = self.opts.length

    def __stationarity_test(self, ndarrary):
        # Adjusted Dickey-Fuller test for stationarity
        #       H0: non-stationarity (linear trends at least)
        #       H1: stationarity (linear trend at least)
        p = adfuller(ndarrary, autolag="AIC")[1]
        if p>=0.05:
            print(f"Time series was not stationary (p={p} Adjusted Dickey-Fullet test using AIC). \n It will be made stationary: out[i]=ndarrary[i+1]-ndarray[i]! Be sure this is what you want...")
            ndarrary = np.diff(ndarrary)
        else:
            return ndarrary

    def fit_subject(
            self, subject_file, run_self_loops=False, make_stationary=False, verbose=True
        ):
        """
        TODO: Add description of the function

        Arguments
        -----------
        subject_file: (string) Full path to the file containing the time series. ROI time series are stored as columns.
        TODO: finish arguments

        Outputs
        -----------
        TODO: Add output description.
        """
        
        name_subject = subject_file.split("/")[-1].split("_TS")[0] + '_Length-' + str(self.length) + '_Method-bivGC'
        print(f"Participant ID: {name_subject}")
        if verbose:
            print("Loading data")

        # Load time series from subject -- dims: time-points X total-ROIs
        time_series = np.genfromtxt(subject_file, delimiter='\t') 
        if np.isnan(time_series[:,0]).all():
            time_series = time_series[:,1:] # First column is dropped due to Nan
        limit = int(time_series.shape[0]*0.01*self.length)

        # ROIs from input command
        self.ROIs = list(range(time_series.shape[-1])) if self.opts.rois[0] == -1 else [roi-1 for roi in self.opts.rois]
        self.ROIs = sorted(self.ROIs)

        # Time series to analyse -- dims: ROIs X 1 X time-points
        TS2analyse = np.expand_dims(
            np.array([time_series[:limit,roi] for roi in self.ROIs]), axis=1
        )
        if verbose:
            print("Done!")
            print("-----")
            print("Computing bivariate Granger influence")

        # Compute GC causality
        for i, roi_i in enumerate(self.ROIs):
            for j in range(i if run_self_loops else i+1, len(self.ROIs)):
                roi_j = self.ROIs[j]

                # In theory, bivariate GC should only be used for stationary time series
                # We can implement a work around, but it is deactivated by default because
                #       it's not clear this is the correct solution
                if make_stationary:
                    data_i = self.__stationarity_test(TS2analyse[i,0,:])
                    data_j = self.__stationarity_test(TS2analyse[j,0,:])
                else:
                    data_i = TS2analyse[i,0,:]
                    data_j = TS2analyse[j,0,:]                    
                
                # Data in the correct format -- dims: time-points X 2
                # From the docs: The data for testing whether the time series in the second column 
                # Granger causes the time series in the first column.
                data_i2j = np.array([data_j, data_i]).T # From i-->j (j = Aj + Bi)
                data_j2i = np.array([data_i, data_j]).T # From i-->j (i = Ai + Bj)                    
                
                # Bivariate GC Scores
                if verbose:
                    print(f"Estimating the directionality for ROIs [{roi_i},{roi_j}]")
                R_i2j, R_j2i, evidence_i2j, evidence_j2i, Score_i2j, Score_j2i = directionality_test_GC(
                    data_i2j, data_j2i, self.lags, significance=0.05, test='F'
                )
                
                if verbose:
                    print("Done!")
                    print("-----")

                # Generate report --> NO surrogates, and bidirectional influences
                    print(f"Saving the summary for ROIs [{roi_i},{roi_j}]")
                generate_report(
                    self.output_dir, name_subject, roi_i, roi_j,
                    -self.lags, R_i2j, R_j2i, R_i2j*0, R_j2i*0,
                    Score_i2j, Score_j2i, Score_i2j*0, 
                    evidence_i2j, evidence_j2i, evidence_i2j*0
                )

                if verbose:
                    print("Done!")
                    print("-----")
        print("Subject finished!")
        print("-------------------------------")
        return name_subject
    
    def fit_dataset(
            self, run_self_loops=False, make_stationary=False
        ):
        """
        TODO: Add description of the function

        Arguments
        -----------
        subject_file: (string) Full path to the file containing the time series. ROI time series are stored as columns.
        TODO: finish arguments

        Outputs
        -----------
        TODO: Add output description.
        """

        name_subjects = []
        if self.opts.num_jobs == 1:
            print("============= Sequential processing =================")
            print("INFO: \t Parallel or sequential processing depends on the input arguments --num_jobs")
            for f in self.files:
                name_subjects.append(
                    self.fit_subject(f, run_self_loops=run_self_loops, make_stationary=make_stationary, verbose=False)
                )
        else:
            print("============== Parallel processing ==================")
            name_subjects = Parallel(n_jobs=self.opts.num_jobs)(
                delayed(self.fit_subject)(f, run_self_loops=run_self_loops, make_stationary=make_stationary, verbose=False)
                for f in self.files
            )
    
if __name__ == '__main__':
    pass
