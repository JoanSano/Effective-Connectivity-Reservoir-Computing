import numpy as np

## Relative imports
from methods.utils import RCC_average, directionality_test
from methods.reservoir_networks import return_reservoir_blocks
from utils.surrogates.surrogate_tools import create_surrogates, surrogate_reservoirs
from analysis.utils import generate_report, process_subject_summary

class RCC():
    def __init__(self, output_dir, opts, json_file_config) -> None:
        self.output_dir = output_dir
        self.opts = opts
        
        # Initialization of the Reservoir blocks
        self.json_file_config = json_file_config
        self.I2N, self.N2N = return_reservoir_blocks(
            json_file=self.json_file_config, exec_args=self.opts
        )

        # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
        self.lags = np.arange(self.opts.min_lag, self.opts.max_lag)
        
        # Load RCC config 
        self.length, self.ROIs, self.split = self.opts.length, self.opts.rois, self.opts.split,
        self.skip, self.runs, self.N_surrogates =  self.opts.skip, self.opts.runs, self.opts.num_surrogates

    def fit(
            self, subject_file, run_self_loops=False, format='svg', factor=10, verbose=False
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
        if verbose:
            print("++++++++++++++++++++++++++++++++") 
            print(f"Participant ID: {name_subject}")
            print("-------------------------------")
            print("Loading data")

        # Load time series from subject -- dims: time-points X total-ROIs
        time_series = np.genfromtxt(subject_file, delimiter='\t')
        if np.isnan(time_series[:,0]).all():
            time_series = time_series[:,1:] # First column is dropped due to Nan
        limit = int(time_series.shape[0]*0.01*self.length)

        # ROIs from input command
        self.ROIs = list(range(time_series.shape[-1])) if self.ROIs[0] == -1 else [roi-1 for roi in self.ROIs]
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
            print("-------------------------------")
            print("Computing reservoir scores and evidence")

        # Compute RCC causality
        run_self_loops = False
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
                    print("-------------------------------")
                
                # IAAFT surrogates test
                    print(f"Training surrogate reservoirs for ROIs [{roi_i},{roi_j}]")
                surrogate_x2y, surrogate_y2x = surrogate_reservoirs(
                    TS2analyse[i], TS2analyse[j], self.N_surrogates, 
                    self.lags, self.I2N, self.N2N, self.split, self.skip, 
                    surrogate_population
                )  
                if verbose:
                    print("Done!")
                    print("-------------------------------")

                # RCC Scores
                if verbose:
                    print(f"Estimating the directionality for ROIs [{roi_i},{roi_j}]")
                evidence_xy, evidence_x2y, evidence_y2x, Score_xy, Score_x2y, Score_y2x = directionality_test(
                    x2y, y2x, surrogate_x2y, surrogate_y2x, self.lags, 
                    significance=0.05, permutations=False, axis=1, bonferroni=True
                )
                if verbose:
                    print("Done!")
                    print("-------------------------------")

                # Generate report
                    print(f"Saving the summary for ROIs [{roi_i},{roi_j}]")
                generate_report(
                    self.output_dir, name_subject, roi_i, roi_j,
                    self.lags, x2y, y2x, surrogate_x2y, surrogate_y2x,
                    Score_x2y, Score_y2x, Score_xy, 
                    evidence_x2y, evidence_y2x, evidence_xy,
                    plot=[self.opts.plot, format]
                )
                if verbose:
                    print("Done!")
                    print("-------------------------------")
        if verbose:
            print("Subject finished!")
            print("-------------------------------")
        return name_subject

if __name__ == '__main__':
    pass
