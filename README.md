# Effective-Connectivity-Reservoir-Computing

### Overall help
$python main_RCCausality.py --help

usage:
Compute Reservoir Computing Causality on time series. In development.

[-h] [-rf R_FOLDER] [-j NUM_JOBS] [-b {vanilla,sequential,parallel}] [-nb NUM_BLOCKS] [--batch_analysis]
{fmri,logistic} ...

positional arguments:

{fmri,logistic}

  fmri                Analyse fMRI time series; Use the flag [(-h,--help) HELP] to see optional inputs
  logistic            Ignore

options:

-h, --help  show this help message and exit \n
-rf R_FOLDER, --r_folder R_FOLDER  Output directory where results will be stored
-j NUM_JOBS, --num_jobs NUM_JOBS  Number of parallel jobs to launch
-b {vanilla,sequential,parallel}, --blocks {vanilla,sequential,parallel}  Choose the type of architecture
-nb NUM_BLOCKS, --num_blocks NUM_BLOCKS  If not 'vanilla' specifiy as a second argument the number of blocks
--batch_analysis      Train the reservoirs on a batch of time series instead of single training. If not present, a different reservoir will be trained for each time series and the results will be avraged.

### fmri help
$python main_RCCausality.py fmri --help
usage: Compute Reservoir Computing Causality on time series.
In development. fmri [-h] [--dir DIR] [--subjects [SUBJECTS ...]] [--rois ROIS [ROIS ...]] [--split SPLIT]
                                                                                  [--skip SKIP]

options:
  -h, --help            show this help message and exit
  --dir DIR             Relative path pointing to the directory where the data is stored
  --subjects [SUBJECTS ...]
                        List of subjects to process. Default is all. Type -1 for all.
  --rois ROIS [ROIS ...]
                        Space separated list of ROIs to analyse. Set to -1 for whole brain analysis. Default is -1
  --split SPLIT         Train-test split percentage as an integer from 0 to 100
  --skip SKIP           Number of time points to skip when testing predictability