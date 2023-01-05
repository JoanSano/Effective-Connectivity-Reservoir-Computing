import numpy as np
import os

## Relative imports
from execution.RCC_utils import RCC_average
from execution.reservoir_networks import return_reservoir_blocks
from utils.plotting_utils import plot_RCC_input2output

def process_single_subject(subject_file, opts, output_dir, json_file_config, format='svg'):
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

    name_subject = subject_file.split("/")[-1].split("_TS")[0]
    print(f"Participant ID: {name_subject}")
    ROIs, split, skip, runs = opts.rois, opts.split, opts.skip, opts.runs
    
    # Load time series from subject -- dims: time-points X total-ROIs
    time_series = np.genfromtxt(subject_file, delimiter='\t')[:,1:]

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse
    TS2analyse = np.expand_dims(
        np.array([time_series[:,roi] for roi in ROIs]), axis=1
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
            
            # Run RCC on axis #1 (i.e., the time points)
            mean_x2y, sem_x2y, mean_y2x, sem_y2x, _, _ = RCC_average(
                TS2analyse[i], TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=runs
            )
            
            # Destination directories and names of outputs
            output_dir_subject = os.path.join(output_dir,name_subject)
            if not os.path.exists(output_dir_subject):
                os.mkdir(output_dir_subject)
            name_subject_RCC = name_subject + '_RCC_rois-' +str(roi_i+1) + 'vs' + str(roi_j+1)
            name_subject_RCC_figure = os.path.join(output_dir_subject,name_subject_RCC+'.' + format)

            # Plot Causality  
            plot_RCC_input2output(
                lags, mean_x2y, mean_y2x, 
                error_i2o=sem_x2y, error_o2i=sem_y2x, 
                save=name_subject_RCC_figure, dpi=300, 
                series_names=(f'{roi_i+1}', f'{roi_j+1}'), scale=0.720
            )
            

def process_multiple_subjects(subjects_files, opts, output_dir, json_file_config, format='svg'):
    """
    TODO: Add description of the function

    Arguments
    -----------
    TODO: Add arguments 

    Outputs
    -----------
    TODO: Add output description.
    """

    ROIs, split, skip, runs = opts.rois, opts.split, opts.skip, opts.runs
    
    # Load time series from subject -- dims: subjects X time-points X total-ROIs
    time_series = np.array([np.genfromtxt(subject, delimiter='\t')[:,1:] for subject in subjects_files])

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse -- dims: ROIs X subjects X time-points
    TS2analyse = np.array([time_series[...,roi] for roi in ROIs])

    # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
    lags = np.arange(-30,31)
    
    # Initialization of the Reservoir blocks
    I2N, N2N = return_reservoir_blocks(json_file=json_file_config, exec_args=opts)

    # Compute RCC causality 
    run_self_loops = True
    for i, roi_i in enumerate(ROIs):
        for j in range(i if run_self_loops else i+1, len(ROIs)):
            roi_j = ROIs[j]            

            # Run RCC on axis #0 (i.e., the subjects)
            mean_x2y, sem_x2y, mean_y2x, sem_y2x, _, _ = RCC_average( # Dimensions: subjects X time-points
                    TS2analyse[i], TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=0, runs=runs
            )

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
