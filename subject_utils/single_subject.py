import numpy as np

## Relative imports
from utils.RCC_utils import RCC_statistics

def process_subject(subject_file, ROIs=-1):
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
    
    time_series = np.genfromtxt(subject_file, delimiter='\t')
    r43, r44 = time_series[:,42], time_series[:,44]

if __name__ == '__main__':
    pass