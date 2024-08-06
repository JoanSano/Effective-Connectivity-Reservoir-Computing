import numpy as np
from joblib import Parallel, delayed
import os
from statsmodels.tsa.stattools import adfuller

## Relative imports
from methods.utils import directionality_test_GC, directionality_test_pwCGC, directionality_test_spwcgc
from analysis.utils import generate_report
from utils.handle_arguments import initialize_and_grep_files  
from utils.surrogates.surrogate_tools import create_surrogates

class pwGC():
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
            print("Computing pairwise Granger influence")

        # Compute GC causality
        for i, roi_i in enumerate(self.ROIs):
            for j in range(i if run_self_loops else i+1, len(self.ROIs)):
                roi_j = self.ROIs[j]

                # Pairwise GC should only be used for stationary time series
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

                # Generate report --> NO surrogates, nor bidirectional influences
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

        print("INFO: Parallel or sequential processing depends on the input arguments --num_jobs")
        name_subjects = []
        if self.opts.num_jobs == 1:
            print("============= Sequential processing =================")
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

class pwCGC():
    def __init__(self, args=None) -> None:
        # Loading the configurations and files with time series 
        self.opts, self.files, self.output_dir, json_file_config, self.timeseries_type = initialize_and_grep_files(args=args)
        os.remove(os.path.join(self.output_dir,json_file_config)) # No reservoir needed

       # Lags to test; in this scenario, always negative
        min_lag = np.abs(self.opts.min_lag)
        self.lags = np.arange(1,min_lag+1)
        self.max_order = min_lag

        # Load config 
        self.length = self.opts.length

    def __stationarity_test(self, arrary, N=None):
        # Adjusted Dickey-Fuller test for stationarity
        #       H0: non-stationarity (linear trends at least)
        #       H1: stationarity (linear trend at least)

        if N is None:
            N = sorted(arrary.shape)[0] # By chance, smaller dimension of the array (the biggest is likely to be time samples)
        
        stationary_ndarray = np.copy(arrary)
        for i in range(N):
            p = adfuller(arrary[:,i], autolag="AIC")[1]
            if p>=0.05:
                print(f"Time series {i} was not stationary (p={p} Adjusted Dickey-Fullet test using AIC). \n It will be made stationary: out[i]=ndarrary[i+1]-ndarray[i]! Be sure this is what you want...")
                stationary_ndarray[:,i] = np.diff(arrary[:,i])
        
        return stationary_ndarray
    
    def fit_subject(
            self, subject_file, ic='aic', make_stationary=False, verbose=True
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
        
        name_subject = subject_file.split("/")[-1].split("_TS")[0] + '_Length-' + str(self.length) + '_Method-pwCGC'
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
        
        # Time series to analyse -- dims: time-points X ROIs
        TS2analyse = np.array([time_series[:limit,roi] for roi in self.ROIs]).T
        # Pairwise conditional GC should only be used for stationary time series
        # We can implement a work around, but it is deactivated by default because
        #       it's not clear this is the correct solution
        if make_stationary:
            TS2analyse = self.__stationarity_test(TS2analyse)
        
        if verbose:
            print("Done!")
            print("-----")
            print("Computing pairwise conditional Granger influence")

        # Compute GC causality
        for i, roi_i in enumerate(self.ROIs):
            for j in range(i+1, len(self.ROIs)):
                roi_j = self.ROIs[j]

                # Conditional GC Scores
                if verbose:
                    print(f"Estimating the directionality for ROIs [{roi_i},{roi_j}]")
                R_i2j, R_j2i, evidence_i2j, evidence_j2i, Score_i2j, Score_j2i = directionality_test_pwCGC(
                    TS2analyse, i, j, max_order=self.max_order, ic=ic, significance=0.05, test='chi2'
                )

                if verbose:
                    print("Done!")
                    print("-----")

                # Generate report --> NO surrogates, nor bidirectional influences
                    print(f"Saving the summary for ROIs [{roi_i},{roi_j}]")
                generate_report(
                    self.output_dir, name_subject, roi_i, roi_j,
                    [0], R_i2j, R_j2i, R_i2j*0, R_j2i*0,
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
            self, ic='aic', make_stationary=False
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

        print("INFO: Parallel or sequential processing depends on the input arguments --num_jobs")
        name_subjects = []
        if self.opts.num_jobs == 1:
            print("============= Sequential processing =================")
            for f in self.files:
                name_subjects.append(
                    self.fit_subject(f, ic=ic, make_stationary=make_stationary, verbose=False)
                )
        else:
            print("============== Parallel processing ==================")
            name_subjects = Parallel(n_jobs=self.opts.num_jobs)(
                delayed(self.fit_subject)(f, ic=ic, make_stationary=make_stationary, verbose=False)
                for f in self.files
            )

class Spectral_pwCGC():
    def __init__(self, args=None):        
        # Loading the configurations and files with time series 
        self.opts, self.files, self.output_dir, json_file_config, self.timeseries_type = initialize_and_grep_files(args=args)
        os.remove(os.path.join(self.output_dir,json_file_config)) # No reservoir needed
        self.N_surrogates = self.opts.num_surrogates
        
       # Lags to test; in this scenario, always negative
        min_lag = np.abs(self.opts.min_lag)
        self.lags = np.arange(1,min_lag+1)
        self.max_order = min_lag

        # Load config 
        self.length = self.opts.length

    def __stationarity_test(self, arrary, N=None):
        # Adjusted Dickey-Fuller test for stationarity
        #       H0: non-stationarity (linear trends at least)
        #       H1: stationarity (linear trend at least)

        if N is None:
            N = sorted(arrary.shape)[0] # By chance, smaller dimension of the array (the biggest is likely to be time samples)
        
        stationary_ndarray = np.copy(arrary)
        for i in range(N):
            p = adfuller(arrary[:,i], autolag="AIC")[1]
            if p>=0.05:
                print(f"Time series {i} was not stationary (p={p} Adjusted Dickey-Fullet test using AIC). \n It will be made stationary: out[i]=ndarrary[i+1]-ndarray[i]! Be sure this is what you want...")
                stationary_ndarray[:,i] = np.diff(arrary[:,i])       
        
        return stationary_ndarray
    
    def fit_subject(
            self, subject_file, make_stationary=False, max_order=15, ic='aic', 
            significance=0.05, tol=1e-8, save_surrogates=False, plots=False, verbose=True
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
        
        name_subject = subject_file.split("/")[-1].split("_TS")[0] + '_Length-' + str(self.length) + '_Method-pwCGC'
        print(f"Participant ID: {name_subject}")
        if verbose:
            print("Loading data")
            print("Done!")
            print("-----")

        # Load time series from subject -- dims: time-points X total-ROIs
        time_series = np.genfromtxt(subject_file, delimiter='\t') 
        if np.isnan(time_series[:,0]).all():
            time_series = time_series[:,1:] # First column is dropped due to Nan
        limit = int(time_series.shape[0]*0.01*self.length)

        # ROIs from input command
        self.ROIs = list(range(time_series.shape[-1])) if self.opts.rois[0] == -1 else [roi-1 for roi in self.opts.rois]
        self.ROIs = sorted(self.ROIs)
        self.nROIs = len(self.ROIs)
        
        # Time series to analyse -- dims: time-points X ROIs
        TS2analyse = np.array([time_series[:limit,roi] for roi in self.ROIs]).T
        # Pairwise conditional GC should only be used for stationary time series
        # We can implement a work around, but it is deactivated by default because
        #       it's not clear this is the correct solution
        if make_stationary:
            TS2analyse = self.__stationarity_test(TS2analyse)
        
        # Time series to analyse -- dims: ROIs X 1 X time-points
        TS2analyse = np.expand_dims(
            np.array([TS2analyse[:limit,roi] for roi in self.ROIs]), axis=1
        )
        
        # Create surrogates for testing
        if verbose:
            print("Creating surrogates")
        surrogates = create_surrogates(TS2analyse, self.ROIs, self.N_surrogates, factor=1)

        # Compute causality measure
        if verbose:
            print("Done!")
            print("-----")
            print("Computing spectral pairwise conditional Granger influence")
        (
            (F, f, x_axis), 
            (F_surrs, f_surrs, x_surrs), 
            (p_val_F, evidence_F)
        ) = directionality_test_spwcgc(
            TS2analyse[:,0,:].T, surrogates, self.nROIs, max_order=max_order, ic=ic, significance=significance, tol=tol
        )
        
        for i, roi_i in enumerate(self.ROIs):
            for j in range(i+1, len(self.ROIs)):
                roi_j = self.ROIs[j]

                if verbose:
                    print("Done!")
                    print("-----")

                # Generate report --> NO surrogates, nor bidirectional influences
                    print(f"Saving the summary for ROIs [{roi_i},{roi_j}]")
                numerical, figures = generate_report(
                    self.output_dir, name_subject, roi_i, roi_j,
                    [0], np.array([F[i,j]]), np.array(F[j,i]), np.array([0]), np.array([0]),
                    np.array([p_val_F[i,j]]), np.array([p_val_F[i,j]]), np.array([0]), 
                    np.array([evidence_F[i,j]]), np.array([evidence_F[j,i]]), np.array([0])
                )
                np.save(os.path.join(numerical, "frequency-measure.npy"), {"f":f, "frequency": x_axis}, allow_pickle=True)

                if verbose:
                    print("Done!")
                    print("-----")

        if save_surrogates:
            if verbose:
                print("Saving surrogate files!")
                np.save(os.path.join(numerical, "frequency-measure_surrogates.npy"), {"f":f_surrs, "frequency": x_surrs}, allow_pickle=True)
                np.save(os.path.join(numerical, "averaged-measure_surrogates.npy"), F_surrs, allow_pickle=True)

            if verbose:
                print("Done!")
                print("-----")
        print("Subject finished!")
        print("-------------------------------")
        
        if plots:
            if verbose:
                print("Plotting subject!")
            # TODO: Call the plots from the plotting module

            if verbose:
                print("Done!")
                print("-----")

        return name_subject

        # Drop the summary and figures
    
if __name__ == '__main__':
    pass
