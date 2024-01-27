#!/bin/bash -l

# This script is supposed to be called by a higher level job scheduler to a cluster.
#   It should not be treated as a main shell script.
#   For every submitted job in a cluster, this script will launch the corresponding experiment.

process_arguments() {
  while getopts ":L:J:D:R:" opt; do
    case $opt in
      L)
        LENGTH="$OPTARG"
        ;;
      J)
        JOBS="$OPTARG"
        ;;
      D)
        DIR="$OPTARG"
        ;;
      R)
        RESULTS="$OPTARG"
        ;;
      \?)
        echo "Invalid option: -$OPTARG"
        exit 1
        ;;
      :)
        echo "Option -$OPTARG requires an argument."
        exit 1
        ;;
    esac
  done
  
  # Shifting to get the subject
  shift $((OPTIND-1))
  SUBJECTS="$@"
}

# Call the function with the provided arguments
process_arguments "$@"

# Arguments that should be common amongst experiments
split="70"
skip="5"
runs="20"
rois="-1"

python main_RCCausality.py $DIR -rf $RESULTS -j $JOBS --split $split --skip $skip --length $LENGTH --subjects $SUBJECTS --rois $rois --runs $runs fmri