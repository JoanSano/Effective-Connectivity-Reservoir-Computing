import argparse
import os
import glob
import numpy as np
import pandas as pd

# Arguments to run the population analysis on the desired dataset
# The results NEED to be organized as follows. The names ARE NOT random, all datasets need to be follow the same pattern
parser = argparse.ArgumentParser(f"\nPopulation analysis of Reservoir Computing Causality.\nOf course it needs a folder with the results of each subject in the population.")
parser.add_argument('population_folder', type=str, help="Folder name of the population to analyse. It has to be located in the 'analysis' directory")
parser.add_argument('--data_info', action='store_true', help="Display info about the structure of the dataset to analyse")
parser.add_argument('--length', type=str, default='*', help="Input the length you want to include in the analysis")
opts = parser.parse_args()
opts.population_folder = os.path.join(os.getcwd(), "analysis", opts.population_folder)
if opts.data_info:
    print("Dataset needs to be structured in the following way ['*' is the all character(s) wildcard]:",
"""
GitRepo
    |-> analysis
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

############################################################
# Get the results from all the population for a given length
############################################################
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
nodes = (1 + np.sqrt(1+8*len(rois))) // 2 
directed_networks = np.zeros((len(subjects), len(rois)+1, len(rois)+1))
#print(len(files))
#print(directed_networks.shape)
#print(rois)
#print(subjects)
#print(lengths)
print(databse.keys())
#print(len(databse))
print(results[databse[lengths[0]+"-"+subjects[0]+"_"+rois[0]]].shape)
print(results[databse[lengths[0]+"-"+subjects[0]+"_"+rois[0]]][:,4])