#!/bin/bash -l

folder_spec=$1
jobs=$2
length=$3
subj="${@:4}"
split="70"
skip="10"
runs="20"
surrogates="100"
rois="-1"

# AOMIC Schaeffer 
data_dir="Datasets/AOMIC_PIOP_rest_Schaefer 200"
results_dir="Results_Specs-"$folder_spec"_AOMIC-Schaefer-200_Split-"$split"_Length-"$length
python main_RCCausality.py $data_dir -rf $results_dir -j $jobs --split $split --skip $skip --length $length --subjects $subj --rois $rois --num_surrogates $surrogates --runs $runs fmri

# Netsim
#data_dir="Datasets/Netsim/Sim-15/Timeseries"
#results_dir="Results_B-"$folder_spec"_Netsim-Sim-15_Split-"$split"_Length-"$length
#python main_RCCausality.py $data_dir -rf $results_dir -j $jobs --split $split --skip $skip --length $length --subjects $subj --rois $rois --num_surrogates $surrogates --runs $runs fmri

# Logistic
#rois="1 2"
#python main_RCCausality.py $data_dir -rf $results_dir -j $jobs --split $split --skip $skip --length $length --subjects $subj --rois $rois --num_surrogates $surrogates --runs $runs logistic --generate --lags_x2y 2 --lags_y2x 9 --c_x2y 0.4 --c_y2x 0.05 --samples 20 --noise 0.1 1