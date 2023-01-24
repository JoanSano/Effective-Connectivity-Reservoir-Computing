#!/bin/bash -l

data_dir="Datasets/HCP_motor-task_12-subjects"
results_dir="Results-Test_Single-subject"
subjects="sub-101309_TS"

python main_RCCausality.py --r_folder $results_dir --runs 10 fmri --rois 8 55 --subjects $subjects --dir $data_dir