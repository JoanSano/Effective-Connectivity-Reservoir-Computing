# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1t4Zvi_AfDU2cu6pZYBj_oY6llkskjkn2
"""
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller

""" # Stationarity test (not necessary all the time)
# TODO: Incorporate?
for x in range(n_subjects):
  #Compute the stationary test, change the variable to test the other signals
  dftest = adfuller(supplementary, autolag="AIC") #.loc[:,x]
  #Print the p-values showing <0.05 if the test of stationarity is passed
  print(dftest[1]) """

def GC_single_subject(subject_file, opts, output_dir, format='svg'):
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
    time_series = np.genfromtxt(subject_file, delimiter='\t')[:,1:] # First column is dropped due to Nan)
    limit = int(time_series.shape[0]*0.01*length)

    # ROIs from input command
    ROIs = list(range(time_series.shape[-1])) if ROIs[0] == -1 else [roi-1 for roi in ROIs]

    # Time series to analyse -- dims: ROIs X 1 X time-points
    TS2analyse = np.expand_dims(
    np.array([time_series[:limit,roi] for roi in ROIs]), axis=1
    )

    # Lags to test; in this scenario, always negative
    max_lag = np.abs(opts.max_lag)
    lags = np.arange(1,max_lag+1)

    # Compute GC causality
    run_self_loops = False
    for i, roi_i in enumerate(ROIs):
        for j in range(i if run_self_loops else i+1, len(ROIs)):
            roi_j = ROIs[j]
            
            data_i2j = np.array([TS2analyse[j,0,:], TS2analyse[i,0,:]]).T
            data_j2i = np.array([TS2analyse[i,0,:], TS2analyse[j,0,:]]).T

            Score_i2j, Score_j2i = np.zeros((len(lags),)), np.zeros((len(lags),))
            for t, lag in enumerate(lags):
                # We only check GC one lag at a time to emulate RCC procedures
                pvals = grangercausalitytests(data_i2j, maxlag=[lag], verbose=False)[lag][0]
                Score_i2j[t] = 1 - pvals["ssr_ftest"][1]
                pvals = grangercausalitytests(data_j2i, maxlag=[lag], verbose=False)[lag][0]
                Score_j2i[t] = 1 - pvals["ssr_ftest"][1]

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
            i2jlabel, j2ilabel = str(roi_i+1) + ' --> ' + str(roi_j+1), str(roi_j+1) + ' --> ' + str(roi_i+1)
            results = pd.DataFrame({
                "time-lags": -lags[::-1],
                "GCS " + i2jlabel: Score_i2j[::-1],
                "GCS " + j2ilabel: Score_j2i[::-1]
            })
            results.to_csv(name_subject_RCC_numerical, index=False, sep='\t', decimal='.')

            # Is there any ROI-2-ROI figure you can do? Maybe p-values accross lags?
        