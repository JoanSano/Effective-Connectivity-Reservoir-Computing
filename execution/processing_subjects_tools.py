import numpy as np
from numpy.matlib import repmat
import os
import pandas as pd

## Relative imports
from execution.RCC_utils import RCC_average, directionality_test
from execution.reservoir_networks import return_reservoir_blocks
from utils.surrogate_tools import create_surrogates, surrogate_reservoirs
from utils.summary import generate_report

def process_single_subject(subject_file, opts, output_dir, json_file_config, format='svg', factor=10):
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

    length, ROIs, split, skip, runs, N_surrogates = opts.length, opts.rois, opts.split, opts.skip, opts.runs, opts.num_surrogates
    name_subject = subject_file.split("/")[-1].split("_TS")[0] + '_Length-' + str(length)
    print(f"Participant ID: {name_subject}")
    
    # Load time series from subject -- dims: time-points X total-ROIs
    time_series = np.genfromtxt(subject_file, delimiter='\t')# TODO: Add compatibility, delimiter='\t')
    if np.isnan(time_series[:,0]).all():
        time_series = time_series[:,1:] # First column is dropped due to Nan
    limit = int(time_series.shape[0]*0.01*length)

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse -- dims: ROIs X 1 X time-points
    TS2analyse = np.expand_dims(
        np.array([time_series[:limit,roi] for roi in ROIs]), axis=1
    )
    
    # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
    lags = np.arange(opts.min_lag, opts.max_lag)

    # Initialization of the Reservoir blocks
    I2N, N2N = return_reservoir_blocks(json_file=json_file_config, exec_args=opts)
    
    # Generate and predict surrogates
    if len(ROIs)<=(factor/2 + 1):
        surrogate_population = create_surrogates(TS2analyse, ROIs, N_surrogates, factor=factor)
    else:
        surrogate_population = None

    # Compute RCC causality
    run_self_loops = False
    for i, roi_i in enumerate(ROIs):
        for j in range(i if run_self_loops else i+1, len(ROIs)):
            roi_j = ROIs[j]
            
            # Run RCC by splitting on axis #1 (i.e., the time points)
            x2y, y2x, _, _ = RCC_average(
                TS2analyse[i], TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=runs, average=False
            )
            
            # IAAFT surrogates test
            print(f"Training surrogate reservoirs for ROIs ({roi_i},{roi_j}) ...")
            surrogate_x2y, surrogate_y2x = surrogate_reservoirs(
                TS2analyse[i], TS2analyse[j], N_surrogates, lags, I2N, N2N, split, skip, surrogate_population
            )  

            # RCC Scores
            evidence_xy, evidence_x2y, evidence_y2x, Score_xy, Score_x2y, Score_y2x = directionality_test(
                x2y, y2x, surrogate_x2y, surrogate_y2x, lags, significance=0.05, permutations=False, axis=1, bonferroni=True
            )

            # Generate report
            generate_report(
                output_dir, name_subject, roi_i, roi_j,
                lags, x2y, y2x, surrogate_x2y, surrogate_y2x,
                Score_x2y, Score_y2x, Score_xy, 
                evidence_x2y, evidence_y2x, evidence_xy,
                plot=opts.plot, format=format
            )

if __name__ == '__main__':
    pass
