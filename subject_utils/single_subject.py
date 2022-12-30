import numpy as np
import os

## Relative imports
from utils.RCC_utils import RCC_statistics
from utils.plotting_utils import plot_RCC_input2output

def process_subject(subject_file, ROIs, I2N, N2N, split, skip, runs, output_dir, format='svg'):
    """
    TODO: Add description of the function

    Arguments
    -----------
    subject_file: (string) Full path to the file containing the time series. ROI time series are stored as columns.
    ROIs: (list) Contains the regions in atlas space that one wants to analyse. For the full pairwise analysis set to -1. (default is -1) 
    TODO: finish arguments

    Outputs
    -----------
    TODO: Add output description.
    """
    
    # Load time series from subject
    time_series = np.genfromtxt(subject_file, delimiter='\t')[:,1:]

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse
    TS2analyse = np.array([time_series[:,roi] for roi in ROIs])

    # Lags and number of runs to test for a given subject (Note: the number of runs is not really super important in the absence of noise)
    lags, runs = np.arange(-30,31), runs

    # Compute RCC causality
    correlations_x2y, correlations_y2x, results_x2y, results_y2x = RCC_statistics(TS2analyse[0], TS2analyse[1], lags, runs, I2N, N2N, split=split, skip=skip)

    # Statistics
    mean_x2y, sem_x2y = np.mean(correlations_x2y, axis=0), np.std(correlations_x2y, axis=0)/np.sqrt(runs)
    mean_y2x, sem_y2x = np.mean(correlations_y2x, axis=0), np.std(correlations_y2x, axis=0)/np.sqrt(runs)

    # Destination directories and names
    name = subject_file.split("/")[-1].split("_TS")[0]
    output_dir = os.path.join(output_dir,name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    name = name + '_RCC_rois-' +str(ROIs[0]+1) + 'vs' + str(ROIs[1]+1) + '.' + format
    name = os.path.join(output_dir,name)
    # Plot Causality  
    plot_RCC_input2output(lags, mean_x2y, mean_y2x, error_i2o=sem_x2y, error_o2i=sem_y2x, save=name, dpi=300, series_names=(f'R({ROIs[0]+1})', f'R({ROIs[1]+1})'))

if __name__ == '__main__':
    pass
