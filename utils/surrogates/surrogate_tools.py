import numpy as np
from sklearn.utils import resample

## Relative imports
from utils.surrogates.timeseries_surrogates import refined_AAFT_surrogates
from methods.utils import RCC_average
from tqdm import tqdm

def create_surrogates(time_series, ROIS, N_surrogates, factor=10):
    """
    TODO: Add documentation
    time_series (ROIs X 1 X time-points):
    """
    Size_population_surrogates = N_surrogates * factor
    T = time_series.shape[-1]
    surrogates = np.zeros((len(ROIS),Size_population_surrogates,T))
    for r, roi in enumerate(ROIS):
        for surr_sample in range(Size_population_surrogates):
            surrogates[r,surr_sample,:] = refined_AAFT_surrogates(time_series[r])
    return surrogates

def sample_surrogates(surrogates, N_surrogates):
    """
    TODO: Add documentation
    time_series [numpy_array(#ROIS X 1 X time-points)]:
    surrogates [str, numpy array(#lags X ROIS X populations_surrogates)]:
    """
    # Sample from them without replacement
    population = list(range(surrogates.shape[-1]))
    samples = resample(population, n_samples=N_surrogates, replace=False, stratify=None)
    return surrogates[:,samples,:]

def surrogate_reservoirs(
        time_series_i, time_series_j, N_surrogates,  
        lags, I2N, N2N, split, skip, 
        surrogate_population_ij=None, verbose=True
    ):
    # Sample from the surrogate population
    if surrogate_population_ij is not None: 
        surrogate_sample = sample_surrogates(surrogate_population_ij, N_surrogates) # dims = (2 X N_surr X T)
        surrogate_i, surrogate_j = surrogate_sample[0], surrogate_sample[1] # dims = (N_surr X T) each one

    # Predictions and generation of surrogate reservoirs
    surrogate_x2y = np.zeros((len(lags),1,N_surrogates))
    surrogate_y2x = np.zeros((len(lags),1,N_surrogates))
    to_iterate = tqdm(range(N_surrogates)) if verbose else range(N_surrogates)
    for surr in to_iterate:
        # Generate surrogate time series
        if surrogate_population_ij is None: 
            surrogate_i = refined_AAFT_surrogates(time_series_i)
            surrogate_j = refined_AAFT_surrogates(time_series_j)
        # Use the sampled surroggates
        else:
            surrogate_i = np.expand_dims(surrogate_sample[0,surr,:], axis=0) # dims = (1 X T) 
            surrogate_j = np.expand_dims(surrogate_sample[1,surr,:], axis=0) # dims = (1 X T) 
        
        # Predict the surrogates from the real data
        surrogate_x2y[...,surr], _, _, _ = RCC_average(
            time_series_i, surrogate_j, lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=None, average=False
        )
        _, surrogate_y2x[...,surr], _, _ = RCC_average(
            surrogate_i, time_series_j, lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=None, average=False
        )
    return np.squeeze(surrogate_x2y, axis=1), np.squeeze(surrogate_y2x)   