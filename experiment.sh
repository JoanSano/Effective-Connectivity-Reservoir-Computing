#!/bin/bash -l

data_dir="Datasets/HCP_motor-task_12-subjects"

for L in 30 40 50 60 70 80 90 100
do
    results_dir="Results_HCP-Dataset_Length-"$L
    python main_RCCausality.py --num_jobs 4 --r_folder $results_dir --runs 50 fmri --rois -1 --subjects -1 --dir $data_dir --length
done