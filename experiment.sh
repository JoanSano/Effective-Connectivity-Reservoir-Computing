#!/bin/bash -l

jobs="5"
split="100"
skip="10"

folder_spec=$1
length=$2
subj="${@:3}"
runs="20"
surrogates="100"
rois="-1"

# Netsim
data_dir="Datasets/Netsim/Sim-15/Timeseries"
results_dir="Results_B-"$folder_spec"_Netsim-Sim-15_Split-"$split"_Length-"$length
python main_RCCausality.py $data_dir -rf $results_dir -j $jobs --split $split --skip $skip --length $length --subjects $subj --rois $rois --num_surrogates $surrogates --runs $runs fmri

# Logistic
#rois="1 2"
#python main_RCCausality.py $data_dir -rf $results_dir -j $jobs --split $split --skip $skip --length $length --subjects $subj --rois $rois --num_surrogates $surrogates --runs $runs logistic --generate --lags_x2y 2 --lags_y2x 9 --c_x2y 0.4 --c_y2x 0.05 --samples 20 --noise 0.1 1