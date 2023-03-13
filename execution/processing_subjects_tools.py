import numpy as np
from numpy.matlib import repmat
import os
import pandas as pd

## Relative imports
from execution.RCC_utils import RCC_average, directionality_test
from execution.reservoir_networks import return_reservoir_blocks
from utils.timeseries_surrogates import refined_AAFT_surrogates
from utils.plotting_utils import plot_RCC_Evidence

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

    length, ROIs, split, skip, runs, N_surrogates = opts.length, opts.rois, opts.split, opts.skip, opts.runs, opts.num_surrogates
    name_subject = subject_file.split("/")[-1].split("_TS")[0] + '_Length-' + str(length)
    print(f"Participant ID: {name_subject}")
    
    # Load time series from subject -- dims: time-points X total-ROIs
    time_series = np.genfromtxt(subject_file)# TODO: Add compatibility, delimiter='\t')
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
    lags = np.arange(-30,31)

    # Initialization of the Reservoir blocks
    I2N, N2N = return_reservoir_blocks(json_file=json_file_config, exec_args=opts)

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
            surrogate_x2y, surrogate_y2x = np.zeros((len(lags),1,N_surrogates)), np.zeros((len(lags),1,N_surrogates))
            for surr in range(N_surrogates):
                surrogate_i, surrogate_j = refined_AAFT_surrogates(TS2analyse[i]), refined_AAFT_surrogates(TS2analyse[j])
                surrogate_x2y[...,surr], _, _, _ = RCC_average(
                    TS2analyse[i], surrogate_j, lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=None, average=False
                )
                _, surrogate_y2x[...,surr], _, _ = RCC_average(
                    surrogate_i, TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=1, runs=None, average=False
                )
            surrogate_x2y, surrogate_y2x = np.squeeze(surrogate_x2y, axis=1), np.squeeze(surrogate_y2x)

            # Means and Standard Errors of the Mean
            mean_x2y, sem_x2y = np.mean(x2y, axis=1), np.std(x2y, axis=1) / np.sqrt(x2y.shape[1])
            mean_y2x, sem_y2x = np.mean(y2x, axis=1), np.std(y2x, axis=1) / np.sqrt(y2x.shape[1])
            mean_x2ys, sem_x2ys = np.mean(surrogate_x2y, axis=1), np.std(surrogate_x2y, axis=1) / np.sqrt(surrogate_x2y.shape[1])
            mean_y2xs, sem_y2xs = np.mean(surrogate_y2x, axis=1), np.std(surrogate_y2x, axis=1) / np.sqrt(surrogate_y2x.shape[1])

            # RCC Scores
            evidence_xy, evidence_x2y, evidence_y2x, Score_xy, Score_x2y, Score_y2x = directionality_test(
                x2y, y2x, surrogate_x2y, surrogate_y2x, lags, significance=0.05, permutations=False, axis=1, bonferroni=True
            )
            
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
            x2ylabel, y2xlabel = str(roi_i+1) + ' --> ' + str(roi_j+1), str(roi_j+1) + ' --> ' + str(roi_i+1)
            xylabel = str(roi_i+1) + ' <--> ' + str(roi_j+1)
            results = pd.DataFrame({
                "time-lags": lags,
                "RCCS " + xylabel: Score_xy,
                "RCCS " + x2ylabel: Score_x2y,
                "RCCS " + y2xlabel: Score_y2x,
                x2ylabel: mean_x2y,
                y2xlabel: mean_y2x,
                'SEM ' + x2ylabel: sem_x2y,
                'SEM ' + y2xlabel: sem_y2x,
                'Surrogate' + x2ylabel: mean_x2ys,
                'Surrogate' + y2xlabel: mean_y2xs,
                'Surrogate' + 'SEM ' + x2ylabel: sem_x2ys,
                'Surrogate' + 'SEM ' + y2xlabel: sem_y2xs
            })
            results.to_csv(name_subject_RCC_numerical, index=False, sep='\t', decimal='.')

            # Plot Evidence for Causality  
            if opts.plots.lower() == "true" :
                plot_RCC_Evidence(
                    lags,
                    {"data": mean_x2y, "error": sem_x2y, "label": r"$\rho_{\tau}$"+f"({str(roi_i+1)},{str(roi_j+1)})", "color": "darkorange", "style": "-", "linewidth": 1, "alpha": 1}, 
                    {"data": mean_y2x, "error": sem_y2x, "label": r"$\rho_{\tau}$"+f"({str(roi_j+1)},{str(roi_i+1)})", "color": "green", "style": "-", "linewidth": 1, "alpha": 1}, 
                    {"data": mean_x2ys, "error": sem_x2ys, "label": r"$\rho_{\tau}$"+f"({str(roi_i+1)},{str(roi_j+1)}"+r"$_{S}$"+")", "color": "bisque", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
                    {"data": mean_y2xs, "error": sem_y2xs, "label": r"$\rho_{\tau}$"+f"({str(roi_j+1)},{str(roi_i+1)}"+r"$_{S}$"+")", "color": "lightgreen", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
                    save=name_subject_RCC_figure, dpi=300, y_label="Scores", x_label=r"$\tau$"+"(s)", limits=(0,1), #scale=0.720, 
                    significance_marks=[
                        {"data": evidence_x2y, "color": "blue", "label": x2ylabel},
                        {"data": evidence_y2x, "color": "red", "label": y2xlabel},
                        {"data": evidence_xy, "color": "purple", "label": xylabel}
                    ]
                )
            

def process_multiple_subjects(subjects_files, opts, output_dir, json_file_config, format='svg', name_subject=None):
    """
    TODO: Add description of the function. 

    Arguments
    -----------
    TODO: Add arguments 

    Outputs
    -----------
    TODO: Add output description.
    """
    
    length, ROIs, split, skip, runs, N_surrogates = opts.length, opts.rois, opts.split, opts.skip, opts.runs, opts.num_surrogates
    if not name_subject:
        name_subject = subjects_files[-1].split("/")[-1].split("_TS")[0] + '_Length-' + str(length)
    # Load time series from subject -- dims: subjects X time-points X total-ROIs
    # TODO: Add compatibility with files without NaNs in the first colum
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
    run_self_loops = False
    for i, roi_i in enumerate(ROIs):
        for j in range(i if run_self_loops else i+1, len(ROIs)):
            roi_j = ROIs[j]            

            # Run RCC by splitting on axis #0 (i.e., the subjects)
            x2y, y2x, _, _ = RCC_average(
                TS2analyse[i], TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=0, runs=runs, average=False
            )

            # IAAFT surrogates test
            surrogate_x2y, surrogate_y2x = np.zeros((len(lags), 1, N_surrogates)), np.zeros((len(lags), 1, N_surrogates))
            for surr in range(N_surrogates):
                surrogate_i, surrogate_j = refined_AAFT_surrogates(TS2analyse[i]), refined_AAFT_surrogates(TS2analyse[j])
                surrogate_x2y[...,surr], _, _, _ = RCC_average(
                    TS2analyse[i], surrogate_j, lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=0, runs=None, average=True
                )
                _, surrogate_y2x[...,surr], _, _ = RCC_average(
                    surrogate_i, TS2analyse[j], lags, I2N, N2N, split=split, skip=skip, shuffle=False, axis=0, runs=None, average=True
                )
            surrogate_x2y, surrogate_y2x = np.squeeze(surrogate_x2y, axis=1), np.squeeze(surrogate_y2x)
            
            # Means and Standard Errors of the Mean
            mean_x2y, sem_x2y = np.mean(x2y, axis=1), np.std(x2y, axis=1) / np.sqrt(x2y.shape[1])
            mean_y2x, sem_y2x = np.mean(y2x, axis=1), np.std(y2x, axis=1) / np.sqrt(y2x.shape[1])
            mean_x2ys, sem_x2ys = np.mean(surrogate_x2y, axis=1), np.std(surrogate_x2y, axis=1) / np.sqrt(surrogate_x2y.shape[1])
            mean_y2xs, sem_y2xs = np.mean(surrogate_y2x, axis=1), np.std(surrogate_y2x, axis=1) / np.sqrt(surrogate_y2x.shape[1])

            # RCC Scores
            evidence_xy, evidence_x2y, evidence_y2x, Score_xy, Score_x2y, Score_y2x = directionality_test(
                x2y, y2x, surrogate_x2y, surrogate_y2x, lags, significance=0.05, permutations=False, axis=1, bonferroni=True
            )
            
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
            x2ylabel, y2xlabel = str(roi_i+1) + ' --> ' + str(roi_j+1), str(roi_j+1) + ' --> ' + str(roi_i+1)
            xylabel = str(roi_i+1) + ' <--> ' + str(roi_j+1)
            results = pd.DataFrame({
                "time-lags": lags * 0.720,
                "RCCS " + xylabel: Score_xy,
                "RCCS " + x2ylabel: Score_x2y,
                "RCCS " + y2xlabel: Score_y2x,
                x2ylabel: mean_x2y,
                y2xlabel: mean_y2x,
                'SEM ' + x2ylabel: sem_x2y,
                'SEM ' + y2xlabel: sem_y2x,
                'Surrogate' + x2ylabel: mean_x2ys,
                'Surrogate' + y2xlabel: mean_y2xs,
                'Surrogate' + 'SEM ' + x2ylabel: sem_x2ys,
                'Surrogate' + 'SEM ' + y2xlabel: sem_y2xs
            })
            results.to_csv(name_subject_RCC_numerical, index=False, sep='\t', decimal='.')

            # Plot Evidence for Causality  
            plot_RCC_Evidence(
                lags,
                {"data": mean_x2y, "error": sem_x2y, "label": r"$\rho_{\tau}$"+f"({str(roi_i+1)},{str(roi_j+1)})", "color": "darkorange", "style": "-", "linewidth": 1, "alpha": 1}, 
                {"data": mean_y2x, "error": sem_y2x, "label": r"$\rho_{\tau}$"+f"({str(roi_j+1)},{str(roi_i+1)})", "color": "green", "style": "-", "linewidth": 1, "alpha": 1}, 
                {"data": mean_x2ys, "error": sem_x2ys, "label": r"$\rho_{\tau}$"+f"({str(roi_i+1)},{str(roi_j+1)}"+r"$_{S}$"+")", "color": "bisque", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
                {"data": mean_y2xs, "error": sem_y2xs, "label": r"$\rho_{\tau}$"+f"({str(roi_j+1)},{str(roi_i+1)}"+r"$_{S}$"+")", "color": "lightgreen", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
                save=name_subject_RCC_figure, dpi=300, y_label="Scores", x_label=r"$\tau$"+"(s)", limits=(0,1), #scale=0.720, 
                significance_marks=[
                    {"data": evidence_x2y, "color": "blue", "label": x2ylabel},
                    {"data": evidence_y2x, "color": "red", "label": y2xlabel},
                    {"data": evidence_xy, "color": "purple", "label": xylabel}
                ]
            )

if __name__ == '__main__':
    pass
