import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# Relative imports
from analysis_utils import *

# Arguments to run the population analysis on the desired dataset
# The results NEED to be organized as follows. The names ARE NOT random, all datasets need to be follow the same pattern
parser = argparse.ArgumentParser(f"\nPopulation analysis of Reservoir Computing Causality.\nOf course it needs a folder with the results of each subject in the population.")
parser.add_argument('population_folder', type=str, help="Folder name of the population to analyse. It has to be located in the 'analysis' directory")
parser.add_argument('figures_folder', type=str, help="Folder name where the figures and results will be stored")
parser.add_argument('method', type=str, help="Method used to test directionality")
parser.add_argument('--data_info', action='store_true', help="Display info about the structure of the dataset to analyse")
parser.add_argument('--length', type=str, default='*', help="Input the length you want to include in the analysis")
opts = parser.parse_args()
opts.population_folder = os.path.join(os.getcwd(), "Results-to-Analyse", opts.population_folder)
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

##########################
# Performance evaluation #
##########################
# Scores for each tested time lag
sensitivity = np.zeros((len(lengths), len(subjects), N_lags))
specificity = np.zeros((len(lengths), len(subjects), N_lags))
pos_pred_value = np.zeros((len(lengths), len(subjects), N_lags))
neg_pred_value = np.zeros((len(lengths), len(subjects), N_lags))
auc = np.zeros((len(lengths), len(subjects), N_lags))
true_positive_rate = np.empty(shape=(len(lengths), len(subjects), N_lags), dtype=object)
false_positive_rate = np.empty(shape=(len(lengths), len(subjects), N_lags), dtype=object)

# Score that takes into account the overall lags
global_sensitivity = np.zeros((len(lengths), len(subjects)))
global_specificity = np.zeros((len(lengths), len(subjects)))
global_pos_pred_value = np.zeros((len(lengths), len(subjects)))

#######################
# Population analysis #
#######################
bonferroni, significance = True, 0.05
for i, L in enumerate(lengths):
    directed_weighted_networks = np.zeros( # shape: Subjects X Lags X Nodes X Nodes
        (len(subjects), N_lags, nodes, nodes)
        )
    directed_binary_networks = np.copy(directed_weighted_networks)

    for j, S in enumerate(subjects):

        ### Load Ground Truth
        #####################
        if "Logistic" in opts.population_folder:
            # Customize the ground truth according to your simulations ###
            which_lag = 2
            if which_lag == 2:
                GT = np.array([ 
                    [0,0.4],
                    [0,0]
                ])
                Binary_GT = np.where(GT>0, 1, 0)    # We binarize the network
                Mask_N1 = Binary_GT + Binary_GT.T   # We constrain to 1st neighbours. Measure performance only in direct connections
            elif which_lag == 9:
                GT = np.array([ 
                    [0,0],
                    [0.05,0]
                ])
                Binary_GT = np.where(GT>0, 1, 0)    # We binarize the network
                Mask_N1 = Binary_GT + Binary_GT.T   # We constrain to 1st neighbours. Measure performance only in direct connections
            else:
                GT = np.array([ 
                    [0,0.4],
                    [0.05,0]
                ])
                Binary_GT = np.where(GT>0, 1, 0) # We binarize the network
                Mask_N1 = np.copy(Binary_GT)     # We constrain to 1st neighbours. Measure performance only in direct connections
        elif "NetSim" in opts.population_folder:
            sim = 15
            gt_path = os.path.join(os.getcwd(), f"Datasets/Netsim/Sim-{sim}/Networks/{S}_sim-{sim}_Net.txt")
            GT = np.genfromtxt(gt_path, delimiter="\t")
            GT += np.eye(GT.shape[0])           # The original diagonal is filled with -1s
            Binary_GT = np.where(GT>0, 1, 0)    # We binarize the network
            Mask_N1 = Binary_GT + Binary_GT.T   # We constrain to 1st neighbours. Measure performance only in direct connections

        else: 
            # HCP Data
            pass

        ### Network reconstruction from RCC Scores
        ##########################################
        for r, Rois in enumerate(rois):
            roi_x = int(Rois.split("-")[-1].split("vs")[0]) - 1
            roi_y = int(Rois.split("-")[-1].split("vs")[1]) - 1
            
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
            else:
                raise NotImplementedError

            for t, tau in enumerate(lags):
                ### We only test for unidirectionality --> We don't consider bidirectional scores
                ######################################
                directed_weighted_networks[j,t,roi_x,roi_y] = Score_x2y[t] # Score_xy[t] + Score_x2y[t] 
                directed_weighted_networks[j,t,roi_y,roi_x] = Score_y2x[t] # Score_xy[t] + Score_y2x[t] 
                directed_binary_networks[j,t,roi_x,roi_y] = evidence_x2y[t] # evidence_xy[t] + evidence_x2y[t]
                directed_binary_networks[j,t,roi_y,roi_x] = evidence_y2x[t] # evidence_xy[t] + evidence_y2x[t]
                
        ### Performance measure
        #######################
        # Lag specific
        for t, tau in enumerate(lags):
            # Masking to first neighbours (i.e., direct connections)
            sensitivity[i,j,t], specificity[i,j,t], pos_pred_value[i,j,t], neg_pred_value[i,j,t] = confusion_matrix_scores(
                Binary_GT, directed_binary_networks[j,t], Mask_N1=Mask_N1
            )
            false_positive_rate[i,j,t], true_positive_rate[i,j,t], auc[i,j,t] = roc_analysis(
                Binary_GT, directed_weighted_networks[j,t], Mask_N1=Mask_N1
            )
        # Overall predictions
        # TODO: Think about how one can incorporate information from all time lags
        global_sensitivity[i,j] = sensitivity[i,j,:].mean(axis=-1)
        global_specificity[i,j] = specificity[i,j,:].mean(axis=-1)
        global_pos_pred_value[i,j] = pos_pred_value[i,j,:].mean(axis=-1)
                
###############
### Figures ###
###############
figures_dir = os.path.join(os.getcwd(), opts.figures_folder)
if not os.path.exists(figures_dir):
    os.mkdir(figures_dir)
style = "solid"
limits = [(0.25,0.75), (0.25,0.75), (0,1), (0,1), (0.5,1)]
colors = ["darkorange", "green", "blue", "red", "black"]

auc_to_plot, ppv_to_plot, sensitivity_to_plot, npv_to_plot, specificity_to_plot = [], [], [], [], []
for i, L in enumerate(lengths):
    auc_to_plot.append(
       {"data": auc[i].mean(axis=0), "error": auc[i].std(axis=0)/np.sqrt(len(subjects)), "label": L, "color": colors[i], "style": style, "linewidth": 0.25, "alpha": 0.75},  
    )
    ppv_to_plot.append(
       {"data": pos_pred_value[i].mean(axis=0), "error": pos_pred_value[i].std(axis=0)/np.sqrt(len(subjects)), "label": L, "color": colors[i], "style": style, "linewidth": 0.25, "alpha": 0.75},  
    )
    sensitivity_to_plot.append(
       {"data": sensitivity[i].mean(axis=0), "error": sensitivity[i].std(axis=0)/np.sqrt(len(subjects)), "label": L, "color": colors[i], "style": style, "linewidth": 0.25, "alpha": 0.75},  
    )
    npv_to_plot.append(
       {"data": neg_pred_value[i].mean(axis=0), "error": neg_pred_value[i].std(axis=0)/np.sqrt(len(subjects)), "label": L, "color": colors[i], "style": style, "linewidth": 0.25, "alpha": 0.75},  
    )
    specificity_to_plot.append(
       {"data": specificity[i].mean(axis=0), "error": specificity[i].std(axis=0)/np.sqrt(len(subjects)), "label": L, "color": colors[i], "style": style, "linewidth": 0.25, "alpha": 0.75},  
    )

# AUC
plot_RCC_Evidence(
    np.where(lags!=0, lags,np.nan),
    *auc_to_plot,
    save=figures_dir+"/auc.png", dpi=300, y_label="AUC", x_label=r"$\tau$"+"(step)", limits=limits[0]
)
# PPV
plot_RCC_Evidence(
    np.where(lags!=0, lags,np.nan),
    *ppv_to_plot,
    save=figures_dir+"/ppv.png", dpi=300, y_label="PPV", x_label=r"$\tau$"+"(step)", limits=limits[1]
)
# Sensitivity
plot_RCC_Evidence(
    np.where(lags!=0, lags,np.nan),
    *sensitivity_to_plot,
    save=figures_dir+"/sensitivity.png", dpi=300, y_label="Sensitivity", x_label=r"$\tau$"+"(step)", limits=limits[2]
)
# NPV
plot_RCC_Evidence(
    np.where(lags!=0, lags,np.nan),
    *npv_to_plot,
    save=figures_dir+"/npv.png", dpi=300, y_label="NPV", x_label=r"$\tau$"+"(step)", limits=limits[3]
)
# Specificity
plot_RCC_Evidence(
    np.where(lags!=0, lags,np.nan),
    *specificity_to_plot,
    save=figures_dir+"/specificity.png", dpi=300, y_label="Specificity", x_label=r"$\tau$"+"(step)", limits=limits[4]
)