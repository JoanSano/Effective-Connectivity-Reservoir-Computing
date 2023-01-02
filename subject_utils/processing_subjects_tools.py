import numpy as np
import os

## Relative imports
from utils.RCC_utils import RCC_statistics
from utils.reservoir_networks import return_reservoir_blocks
from utils.plotting_utils import plot_RCC_input2output

def process_subject(subject_file, opts, output_dir, json_file_config, format='svg'):
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

    ROIs, split, skip, runs = opts.rois, opts.split, opts.skip, opts.runs
    
    # Load time series from subject
    time_series = np.genfromtxt(subject_file, delimiter='\t')[:,1:]

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse
    TS2analyse = np.array([time_series[:,roi] for roi in ROIs])

    # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
    lags, runs = np.arange(-30,31), runs

    # Compute RCC causality
    run_self_loops = True
    for i, roi_i in enumerate(ROIs):
        for j in range(i if run_self_loops else i+1, len(ROIs)):
            roi_j = ROIs[j]
            # Initialization of the Reservoir blocks
            I2N, N2N = return_reservoir_blocks(json_file=json_file_config, exec_args=opts)
            print(TS2analyse[i].shape)
            # Run RCC
            correlations_x2y, correlations_y2x, results_x2y, results_y2x = RCC_statistics(
                TS2analyse[i], TS2analyse[j], lags, runs, I2N, N2N, split=split, skip=skip
            )
            del I2N, N2N

            # Statistics
            mean_x2y, sem_x2y = np.mean(correlations_x2y, axis=0), np.std(correlations_x2y, axis=0)/np.sqrt(runs)
            mean_y2x, sem_y2x = np.mean(correlations_y2x, axis=0), np.std(correlations_y2x, axis=0)/np.sqrt(runs)

            # Destination directories and names of outputs
            name_subject = subject_file.split("/")[-1].split("_TS")[0]
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
                series_names=(f'R({roi_i+1})', f'R({roi_j+1})')
            )

def process_multiple_subjects(subjects_files, opts, output_dir, json_file_config, format='svg'):
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


if __name__ == '__main__':
    pass
