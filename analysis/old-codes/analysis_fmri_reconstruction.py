import argparse
import os
import glob
import numpy as np
import pandas as pd
import json

# Relative imports
from analysis_utils import *

# Arguments to run the population analysis on the desired dataset
# The results NEED to be organized as follows. The names ARE NOT random, all datasets need to be follow the same pattern
parser = argparse.ArgumentParser(f"\nPopulation analysis of Reservoir Computing Causality.\nOf course it needs a folder with the results of each subject in the population.")
parser.add_argument('population_folder', type=str, help="Folder name of the population to analyse. It has to be located in the 'analysis' directory")
parser.add_argument('results_folder', type=str, help="Folder name where the figures and results will be stored")
parser.add_argument('--method', type=str, default="RCC", choices=["RCC", "GC", "other"], help="Method used to test directionality")
parser.add_argument('--data_info', action='store_true', help="Display info about the structure of the dataset to analyse")
parser.add_argument('--length', type=str, default='*', help="Input the length you want to include in the analysis")
parser.add_argument('--rsnet', type=str, default='brain', help="Reconstructed resting-state network")
opts = parser.parse_args()
opts.population_folder = os.path.join(os.getcwd(), "Results-to-Analyse", opts.population_folder)
results_dir = os.path.join(os.getcwd(), opts.results_folder)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)
if opts.data_info:
    print("Dataset needs to be structured in the following way ['*' is the all character(s) wildcard]:",
"""
GitRepo
    |-> Results-to-Analyse
        |-> population_folder
            |-> *Length-AA
            |-> ...
            |-> *Length-ZZ
                |-> sub-yy_*_Length-ZZ
                |-> ..
                |-> sub-xx_*_Length-ZZ
                    |-> Figures
                        |-> sub-xx_*_Length-ZZ_RCC_rois-1vs2.png
                        |-> ...
                        |-> sub-xx_*_Length-ZZ_RCC_rois-1vs2.png
                    |-> Numerical
                        |-> sub-xx_*_Length-ZZ_RCC_rois-1vs2.tsv
                        |-> ...
                        |-> sub-xx_*_Length-ZZ_RCC_rois-1vs2.tsv
"""
    )
    quit()

################################################################
# Get the results from all the population for all given length #
################################################################
files = glob.glob(opts.population_folder+f"/*Length-{opts.length}/*/*/*.tsv")
header = list(pd.read_csv(files[0], sep="\t").keys())
'''
column name                |||  Index value
===========================================
time-lags                   0
RCCS 1 <--> 2               1
RCCS 1 --> 2                2
RCCS 2 --> 1                3
1 --> 2                     4
2 --> 1                     5
SEM 1 --> 2                 6
SEM 2 --> 1                 7
Surrogate1 --> 2            8
Surrogate2 --> 1            9
SurrogateSEM 1 --> 2        10
SurrogateSEM 2 --> 1        11
'''
results = np.empty(shape=len(files), dtype=object)
subjects, rois, lengths, databse = [], [], [], {}
for i, f in enumerate(files):
    results[i] = np.genfromtxt(f, delimiter="\t", skip_header=1)
    length = f.split("/")[-3].split("_")[-1]
    subject_ID = f.split("/")[-3].split("_")[0]
    pair_rois = f.split("/")[-1].split("_")[-1].split(".")[0]
    if subject_ID not in subjects:
        subjects.append(subject_ID)
    if pair_rois not in rois:
        rois.append(pair_rois)
    if length not in lengths:
        lengths.append(length)
    # We create a database with the key and the position of the file
    databse[length+"-"+subject_ID+"_"+pair_rois] = i
# We compute the number of nodes in the network based on the rois loaded
nodes = int((1 + np.sqrt(1+8*len(rois))) // 2)
N_lags = len(results[databse[lengths[0]+"-"+subjects[0]+"_"+rois[0]]][:,0])
N_subjects = len(subjects)
# Roi and indices to save the data to
roi_2_id, keys= {}, []
for _, pair in enumerate(rois):
    roi = int(pair.split("-")[-1].split("vs")[0])
    if roi not in keys:
        keys.append(roi)    
    roi = int(pair.split("-")[-1].split("vs")[1])
    if roi not in keys:
        keys.append(roi)
for k, v in zip(np.sort(np.array(keys, dtype=int)), range(0,len(keys))):
    roi_2_id[str(k)] = v
del keys
with open(f"{opts.results_folder}/Key_Node-ROIs.json", 'w') as f:
    json.dump(roi_2_id, f, indent=2)

##########################
# Performance evaluation #
##########################
# Networks
Weighted_nets = np.empty(shape=len(lengths), dtype=object)
Binary_nets = np.empty(shape=len(lengths), dtype=object)

##########################
# Network Reconstruction #
##########################
bonferroni, significance = True, 0.05
for i, L in enumerate(lengths):
    directed_weighted_networks = np.zeros( # shape: Lags X Nodes X Nodes
        (N_lags, N_subjects, nodes, nodes)
        )
    directed_binary_networks = np.copy(directed_weighted_networks)
    symmetric_weighted_networks = np.copy(directed_weighted_networks)
    symmetric_binary_networks = np.copy(directed_weighted_networks)
    for j, S in enumerate(subjects):
        subject_dir = os.path.join(opts.results_folder, S)
        if not os.path.exists(subject_dir):
            os.mkdir(subject_dir)

        for r, Rois in enumerate(rois):
            roi_x = roi_2_id[str(Rois.split("-")[-1].split("vs")[0])]
            roi_y = roi_2_id[str(Rois.split("-")[-1].split("vs")[1])]
            
            lags = results[databse[L+"-"+S+"_"+rois[r]]][:,0]

            if opts.method == "RCC":
                Score_xy = results[databse[L+"-"+S+"_"+rois[r]]][:,1]
                Score_x2y = results[databse[L+"-"+S+"_"+rois[r]]][:,2]
                Score_y2x = results[databse[L+"-"+S+"_"+rois[r]]][:,3]
                # Significance Threshold
                if bonferroni:
                    # Num hypothesis is 3
                    threshold_uni, _ = unidirectional_score_ij(significance/2, significance/2, significance/2, significance/2, -1)
                    threshold_bi = score_ij(significance/3, significance/3, significance/(2*3))
                else:
                    threshold_uni, _ = unidirectional_score_ij(significance, significance, significance, significance, -1)
                    threshold_bi = score_ij(significance, significance, significance/2)
                evidence_x2y = np.where(Score_x2y>=threshold_uni, 1, 0)
                evidence_y2x = np.where(Score_y2x>=threshold_uni, 1, 0)
                evidence_xy = np.where(Score_xy>=threshold_bi, 1, 0)
            elif opts.method == "GC":
                Score_x2y = results[databse[L+"-"+S+"_"+rois[r]]][:,1]
                Score_y2x = results[databse[L+"-"+S+"_"+rois[r]]][:,2]
                evidence_x2y = np.where(Score_x2y>=(1-significance), 1, 0)
                evidence_y2x = np.where(Score_y2x>=(1-significance), 1, 0)
                # GC does not consider bidirectionality independently
            else:
                raise NotImplementedError
            
            for t, tau in enumerate(lags):
                ### We consider bidirectional and unidirectional scores
                directed_weighted_networks[t,j,roi_x,roi_y] =  Score_x2y[t]
                directed_weighted_networks[t,j,roi_y,roi_x] =  Score_y2x[t]
                directed_binary_networks[t,j,roi_x,roi_y] = evidence_x2y[t]
                directed_binary_networks[t,j,roi_y,roi_x] = evidence_y2x[t]
                if opts.method == "RCC":
                    symmetric_weighted_networks[t,j,roi_x,roi_y] =  Score_xy[t]
                    symmetric_weighted_networks[t,j,roi_y,roi_x] =  Score_xy[t]
                    symmetric_binary_networks[t,j,roi_x,roi_y] =  evidence_xy[t]
                    symmetric_binary_networks[t,j,roi_y,roi_x] =  evidence_xy[t]

        # Saving for each subject    
        for t, tau in enumerate(lags): 
            np.savetxt(
                f"{subject_dir}/{S}_rsNET-{opts.rsnet}_lag={tau}_directed-weighted_{L}.tsv",
                directed_weighted_networks[t,j],
                delimiter="\t"
            )
            np.savetxt(
                f"{subject_dir}/{S}_rsNET-{opts.rsnet}_lag={tau}_directed_{L}.tsv",
                directed_binary_networks[t,j],
                delimiter="\t"
            )
            if opts.method == "RCC":
                np.savetxt(
                    f"{subject_dir}/{S}_rsNET-{opts.rsnet}_lag={tau}_symmetric-weighted_{L}.tsv",
                    symmetric_weighted_networks[t,j],
                    delimiter="\t"
                )
                np.savetxt(
                    f"{subject_dir}/{S}_rsNET-{opts.rsnet}_lag={tau}_symmetric-binary_{L}.tsv",
                    symmetric_binary_networks[t,j],
                    delimiter="\t"
                )

    # Save the mean reconstructed networks for each length
    for t, tau in enumerate(lags):
        np.savetxt(f"{opts.results_folder}/rsNET-{opts.rsnet}_lag={tau}_directed-weighted_{L}.tsv", directed_weighted_networks[t].mean(axis=0), delimiter="\t")
        np.savetxt(f"{opts.results_folder}/rsNET-{opts.rsnet}_lag={tau}_directed-binary_{L}.tsv", directed_binary_networks[t].mean(axis=0), delimiter="\t")
        if opts.method == "RCC":
            np.savetxt(f"{opts.results_folder}/rsNET-{opts.rsnet}_lag={tau}_symmetric-weighted_{L}.tsv", symmetric_weighted_networks[t].mean(axis=0), delimiter="\t")
            np.savetxt(f"{opts.results_folder}/rsNET-{opts.rsnet}_lag={tau}_symmetric-binary_{L}.tsv", symmetric_binary_networks[t].mean(axis=0), delimiter="\t")
                