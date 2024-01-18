import numpy as np
import os
from glob import glob
import pandas as pd 
import json

# Relative imports
from analysis.plotting import plot_evidence

def generate_report(
        directory, name_subject, roi_i, roi_j,
        lags, i2j, j2i, surrogate_i2j, surrogate_j2i,
        Score_i2j, Score_j2i, Score_ij, 
        evidence_i2j, evidence_j2i, evidence_ij,
        plot=[True, 'svg']
    ):   
    """
    Saves a summary file with the following column names and info. The format is .tsv
                    column name                |||  Index value
                    ===========================================
                    time-lags                   0
                    Score 1 <--> 2              1
                    Score 1 --> 2               2
                    Score 2 --> 1               3
                    Evidence 1 <--> 2           4
                    Evidence 1 --> 2            5
                    Evidence 2 --> 1            6
                    1 --> 2                     7
                    2 --> 1                     8
                    SEM 1 --> 2                 9
                    SEM 2 --> 1                 10
                    1 --> 2Surrogate            11
                    2 --> 1Surrogate            12
                    1 --> 2Surrogate SEM        13
                    2 --> 1Surrogate SEM        14
    It can also plot and/or save the results in a comprehensive visual summary.

    Arguments
    ---------
    lags:
    
    Returns
    ---------    
    """         
    # Means and Standard Errors of the Mean
    mean_i2j, sem_i2j = np.mean(i2j, axis=1), np.std(i2j, axis=1) / np.sqrt(i2j.shape[1])
    mean_j2i, sem_j2i = np.mean(j2i, axis=1), np.std(j2i, axis=1) / np.sqrt(j2i.shape[1])
    mean_i2js, sem_i2js = np.mean(surrogate_i2j, axis=1), np.std(surrogate_i2j, axis=1) / np.sqrt(surrogate_i2j.shape[1])
    mean_j2is, sem_j2is = np.mean(surrogate_j2i, axis=1), np.std(surrogate_j2i, axis=1) / np.sqrt(surrogate_j2i.shape[1])

    # Destination directories and names of outputs
    output_dir_subject = os.path.join(directory,name_subject)
    numerical = os.path.join(output_dir_subject,"Numerical")
    figures = os.path.join(output_dir_subject,"Figures")
    if not os.path.exists(output_dir_subject):
        os.mkdir(output_dir_subject)
    if not os.path.exists(numerical):
        os.mkdir(numerical)
    if not os.path.exists(figures):
        os.mkdir(figures)
    plot, format = plot
    name_subject_RCC = name_subject + '_ROIs-' +str(roi_i+1) + 'vs' + str(roi_j+1)
    name_subject_RCC_figure = os.path.join(figures, name_subject_RCC+'.' + format)
    name_subject_RCC_numerical = os.path.join(numerical ,name_subject_RCC+'.tsv')

    # Save numerical results
    i2jlabel, j2ilabel = str(roi_i+1) + ' --> ' + str(roi_j+1), str(roi_j+1) + ' --> ' + str(roi_i+1)
    ijlabel = str(roi_i+1) + ' <--> ' + str(roi_j+1)
    results = pd.DataFrame({
        "time-lags": lags,
        "Score " + ijlabel: Score_ij,
        "Score " + i2jlabel: Score_i2j,
        "Score " + j2ilabel: Score_j2i,
        "Evidence " + ijlabel: evidence_ij,
        "Evidence " + i2jlabel: evidence_i2j,
        "Evidence " + j2ilabel: evidence_j2i,
        i2jlabel: mean_i2j,
        j2ilabel: mean_j2i,
        'SEM ' + i2jlabel: sem_i2j,
        'SEM ' + j2ilabel: sem_j2i,
        i2jlabel + 'Surrogate': mean_i2js,
        j2ilabel + 'Surrogate': mean_j2is,
        'SEM ' + i2jlabel + 'Surrogate': sem_i2js,
        'SEM ' + j2ilabel + 'Surrogate': sem_j2is
    })
    results.to_csv(name_subject_RCC_numerical, index=False, sep='\t', decimal='.')

def process_subject_summary(
        output_dir, name_subject
    ):       
    """
    The summary file, for each pair of ROIs has the following structure:
            column name                |||  Index value
            ===========================================
            time-lags                   0
            Score 1 <--> 2              1
            Score 1 --> 2               2
            Score 2 --> 1               3
            Evidence 1 <--> 2           4
            Evidence 1 --> 2            5
            Evidence 2 --> 1            6
            1 --> 2                     7
            2 --> 1                     8
            SEM 1 --> 2                 9
            SEM 2 --> 1                 10
            1 --> 2Surrogate            11
            2 --> 1Surrogate            12
            1 --> 2Surrogate SEM        13
            2 --> 1Surrogate SEM        14
    Loads the following summary and stores it in an object with a more convenient, but less compact format.

    Arguments
    ---------
    output_dir      (str):
    name_subject    (str):

    Returns
    ---------
    results_subject (numpy array): [pairs, ] containing the database with the results
    header                 (dict): key-value pairs of the header name and the column ID in results_subject
    ROI_CODE_Converter     (dict): Contains the codes to move between 0-indexing of ROIs and edges to the
                                   original numbers (It does not include the label names, only numbers!)
    """

    # Destination directories and names of outputs
    output_dir_subject = os.path.join(output_dir,name_subject)
    numerical = os.path.join(output_dir_subject,"Numerical")
    
    # Get files
    files_interactions = glob(numerical+"/*.tsv")
    header = {k: v for v,k in enumerate(list(pd.read_csv(files_interactions[0], sep="\t").keys()))}
    
    results_subject = np.empty(shape=len(files_interactions), dtype=object)
    key_pairwise, key_ndoe2roi, key_pairwise_0Indexed, v = {}, {}, {}, 0
    for i, f in enumerate(files_interactions):
        # Load summary
        results_subject[i] = np.genfromtxt(f, delimiter="\t", skip_header=1)
        # Each file encodes a pairwise interaction
        pair_rois = f.split("/")[-1].split("_")[-1].split(".")[0].split("-")[-1]
        ri, rj = int(pair_rois.split("vs")[0]), int(pair_rois.split("vs")[1])
        # We add ROIs analyzed starting from 0 because the original dataset doesn't need to start from 0!
        if ri not in key_ndoe2roi.keys():
            key_ndoe2roi[ri] = v
            v += 1
        if rj not in key_ndoe2roi.keys():
            key_ndoe2roi[rj] = v
            v += 1
        # Dictionary with key: value --> 
        key_pairwise_0Indexed[(key_ndoe2roi[ri],key_ndoe2roi[rj])] = i
        key_pairwise[(ri,rj)] = (key_ndoe2roi[ri],key_ndoe2roi[rj])
    
    # We swap because we want the node id as the key and the real ROI as label to plot!
    # There are not repeated values, hence we can do it faster and without carefully checking repeated keys
    key_ndoe2roi = dict([(value, key) for key, value in key_ndoe2roi.items()])

    # Save the code for transparency and other a posteriori plots that the user might want
    # Convert tuple to a string using a method of your choice
    deal_with_tuples_keys = lambda original_dict: {'('+','.join(map(str, k))+')': v for k, v in original_dict.items()}
    deal_with_tuples_value = lambda original_dict: {k: '('+','.join(map(str, v))+')' for k, v in deal_with_tuples_keys(original_dict).items()}
    ROI_CODE_Converter = {
        "0IndexedNode_TO_ROI": key_ndoe2roi,
        "0Indexed_TO_fileID_pairwise_keys": deal_with_tuples_keys(key_pairwise_0Indexed),
        "0Indexed_TO_Original_pariwise_keys":deal_with_tuples_value(key_pairwise)
    }
    with open(f"{output_dir_subject}/Code2Transform_Node2ROIs.json", 'w') as f: 
        json.dump(ROI_CODE_Converter, f, indent=2)
    ROI_CODE_Converter = {
        "0IndexedNode_TO_ROI": key_ndoe2roi,
        "0Indexed_TO_fileID_pairwise_keys": key_pairwise_0Indexed,
        "0Indexed_TO_Original_pariwise_keys":key_pairwise
    }
    return results_subject, header, ROI_CODE_Converter
