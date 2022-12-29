import numpy as np

## Relative imports
from utils.RCC_utils import RCC_statistics

def process_subject(subject_file, ROIs):
    """
    TODO: Add description of the function

    Arguments
    -----------
    subject_file: (string) Full path to the file containing the time series. ROI time series are stored as columns.
    ROIs: (list) Contains the regions in atlas space that one wants to analyse. For the full pairwise analysis set to -1. (default is -1) 

    Outputs
    -----------
    TODO: Add output description.
    """
    
    # Load time series from subject
    time_series = np.genfromtxt(subject_file, delimiter='\t')[:,1:]
    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs == -1 else [roi-1 for roi in ROIs]
    # Time series to analyse
    TS2analyse = np.array([None] * len(ROIs)) # TODO: check what's faster this or an array with predefined length
    for i, roi in enumerate(ROIs):
        TS2analyse[i] = time_series[:,roi]

	# TODO: time lags to test. Adrian suggested to test even further than 2,3 TRs    
if __name__ == '__main__':
    pass
