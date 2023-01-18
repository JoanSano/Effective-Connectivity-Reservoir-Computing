import numpy as np
import os
import pandas as pd

## Relative imports
from execution.RCC_utils import RCC_average
from execution.reservoir_networks import return_reservoir_blocks
from utils.timeseries_surrogates import refined_AAFT_surrogates
from utils.plotting_utils import plot_RCC_input2output

def process_single_subject(subject_file, opts, output_dir, json_file_config, format='svg', N_surrogates=100):
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

    length, ROIs, split, skip, runs = opts.length, opts.rois, opts.split, opts.skip, opts.runs
    name_subject = subject_file.split("/")[-1].split("_TS")[0] + '_Length-' + str(length)
    print(f"Participant ID: {name_subject}")
    
    # Load time series from subject -- dims: time-points X total-ROIs
    time_series = np.genfromtxt(subject_file, delimiter='\t')[:,1:]
    limit = int(time_series.shape[0]*0.01*length)

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse -- dims: ROIs X 1 X time-points
    TS2analyse = np.expand_dims(
        np.array([time_series[:limit,roi] for roi in ROIs]), axis=1
    )
    
    # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
    lags = np.arange(-30,31)

    # Initialization of the Reservoir blocks
    I2N, N2N = return_reservoir_blocks(json_file=json_file_config, exec_args=opts)

    # Compute RCC causality
    run_self_loops = True
    for i, roi_i in enumerate(ROIs):
        for j in range(i if run_self_loops else i+1, len(ROIs)):
            roi_j = ROIs[j]
            
            # Run RCC by splitting on axis #1 (i.e., the time points)
            mean_x2y, sem_x2y, mean_y2x, sem_y2x, _, _ = RCC_average(
                TS2analyse[i], TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=runs
            )

            # IAAFT surrogates test
            surrogate_x2y, surrogate_y2x = np.zeros((len(lags), N_surrogates)), np.zeros((len(lags), N_surrogates))
            for surr in range(N_surrogates):
                surrogate_i, surrogate_j = refined_AAFT_surrogates(TS2analyse[i]), refined_AAFT_surrogates(TS2analyse[i])
                surrogate_x2y[:,surr], _, surrogate_y2x[:,surr], _, _, _ = RCC_average(
                    surrogate_i, surrogate_j, lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=None
                )
            # TODO: Add zscores
            mean_x2y, sem_x2y = np.mean(surrogate_x2y, axis=1), np.std(surrogate_x2y, axis=1)/np.sqrt(N_surrogates)
            mean_y2x, sem_y2x = np.mean(surrogate_y2x, axis=1), np.std(surrogate_y2x, axis=1)/np.sqrt(N_surrogates)

            # Destination directories and names of outputs
            output_dir_subject = os.path.join(output_dir,name_subject)
            numerical = os.path.join(output_dir_subject,"Numerical")
            figures = os.path.join(output_dir_subject,"Figures")
            if not os.path.exists(output_dir_subject):
                os.mkdir(output_dir_subject)
            if not os.path.exists(numerical):
                os.mkdir(numerical)
            if not os.path.exists(figures):
                os.mkdir(figures)
            name_subject_RCC = name_subject + '_RCC_rois-' +str(roi_i+1) + 'vs' + str(roi_j+1)
            name_subject_RCC_figure = os.path.join(figures, name_subject_RCC+'.' + format)
            name_subject_RCC_numerical = os.path.join(numerical ,name_subject_RCC+'.tsv')

            # Save numerical results
            results = pd.DataFrame({
                "time-lags": lags,
                str(roi_i+1) + '-->' + str(roi_j+1): mean_x2y,
                str(roi_j+1) + '-->' + str(roi_i+1): mean_y2x,
                'SEM' + str(roi_i+1) + '-->' + str(roi_j+1): sem_x2y,
                'SEM' + str(roi_j+1) + '-->' + str(roi_i+1): sem_y2x
            })
            results.to_csv(name_subject_RCC_numerical, index=False, sep='\t', decimal='.')

            # Plot Individual Causality  
            plot_RCC_input2output(
                lags, mean_x2y, mean_y2x, 
                error_i2o=sem_x2y, error_o2i=sem_y2x, 
                save=name_subject_RCC_figure, dpi=300, 
                series_names=(f'{roi_i+1}', f'{roi_j+1}'), scale=0.720
            )
            

def process_multiple_subjects(subjects_files, opts, output_dir, json_file_config, format='svg', N_surrogates=100):
    """
    TODO: Add description of the function

    Arguments
    -----------
    TODO: Add arguments 

    Outputs
    -----------
    TODO: Add output description.
    """

    length, ROIs, split, skip, runs = opts.length, opts.rois, opts.split, opts.skip, opts.runs
    
    # Load time series from subject -- dims: subjects X time-points X total-ROIs
    time_series = np.array([np.genfromtxt(subject, delimiter='\t')[:,1:] for subject in subjects_files])
    limit = int(time_series.shape[1]*0.01*length)

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse -- dims: ROIs X subjects X time-points
    TS2analyse = np.array([time_series[:,:limit,roi] for roi in ROIs])

    # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
    lags = np.arange(-30,31)
    
    # Initialization of the Reservoir blocks
    I2N, N2N = return_reservoir_blocks(json_file=json_file_config, exec_args=opts)

    # Compute RCC causality 
    run_self_loops = True
    for i, roi_i in enumerate(ROIs):
        for j in range(i if run_self_loops else i+1, len(ROIs)):
            roi_j = ROIs[j]            

            # Run RCC by splitting on axis #0 (i.e., the subjects)
            mean_x2y, sem_x2y, mean_y2x, sem_y2x, _, _ = RCC_average( # Dimensions: subjects X time-points
                    TS2analyse[i], TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=0, runs=runs
            )

            # IAAFT surrogates test
            surrogate_x2y, surrogate_y2x = np.zeros((len(lags), N_surrogates)), np.zeros((len(lags), N_surrogates))
            for surr in range(N_surrogates):
                surrogate_i, surrogate_j = refined_AAFT_surrogates(TS2analyse[i]), refined_AAFT_surrogates(TS2analyse[i])
                surrogate_x2y[:,surr], _, surrogate_y2x[:,surr], _, _, _ = RCC_average(
                    surrogate_i, surrogate_j, lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=None
                )
            # TODO: Add zscore
            mean_x2y, sem_x2y = np.mean(surrogate_x2y, axis=1), np.std(surrogate_x2y, axis=1)/np.sqrt(N_surrogates)
            mean_y2x, sem_y2x = np.mean(surrogate_y2x, axis=1), np.std(surrogate_y2x, axis=1)/np.sqrt(N_surrogates)

            # Destination names of outputs
            name_roi_RCC = 'RCC_rois-' +str(roi_i+1) + 'vs' + str(roi_j+1)
            name_roi_RCC_figure = os.path.join(output_dir,name_roi_RCC+'.' + format)

            # Plot Causality  
            plot_RCC_input2output(
                lags, mean_x2y, mean_y2x, 
                error_i2o=sem_x2y, error_o2i=sem_y2x, 
                save=name_roi_RCC_figure, dpi=300, 
                series_names=(f'{roi_i+1}', f'{roi_j+1}'), scale=0.720
            )

if __name__ == '__main__':
    pass
