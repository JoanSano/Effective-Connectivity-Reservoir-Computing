# Effective-Connectivity-Reservoir-Computing
I am working on an improved interface, for an easier and more  "plug and play" usage. It will also incorporate bivariate Granger causality and transfer entropy (for now). For urgent requests, please don't hesitate to reach at j.roget@sanoscience.org. Alternatively, take a look at the branch 'update-V1'.


Start by creating a python environment and install the dependencies. Recomended is to use the following: 

```shell
user@user:~/Repository$ conda create -n env-name python=3.10
user@user:~/Repository$ conda activate env-name
user@user:~/Repository$ pip install -r requirements.txt --no-cache-dir
```

Data needs to be stored in a txt file where columns are separated by tabs. The first column of all needs to be full of NaN or empty values. The rest of the columns correspond to the time series of each Region Of Interest (ROI); that is, column 3 contains the time series of ROI number 3.

### To  display needed arguments the main arguments
```shell
user@user:~/Repository$ python main_RCCausality.py --help
usage:
Compute Reservoir Computing Causality on time series.
In development.
 [-h] [-rf R_FOLDER] [-j NUM_JOBS] [-b {vanilla,sequential,parallel}]
                                                                               [-nb NUM_BLOCKS] [--split SPLIT] [--skip SKIP] [--length LENGTH]
                                                                               [--subjects [SUBJECTS ...]] [--rois ROIS [ROIS ...]]
                                                                               [--num_surrogates NUM_SURROGATES] [--batch_analysis | --runs RUNS]
                                                                               dir {fmri,logistic} ...

positional arguments:
  dir                   Relative path pointing to the directory where the data is stored and/or generated
  {fmri,logistic}
    fmri                Analyse fMRI time series; Use the flag [(-h,--help) HELP] to see optional inputs
    logistic            Anlysis of logistic time series to test the method

options:
  -h, --help            show this help message and exit
  -rf R_FOLDER, --r_folder R_FOLDER
                        Output directory where results will be stored
  -j NUM_JOBS, --num_jobs NUM_JOBS
                        Number of parallel jobs to launch
  -b {vanilla,sequential,parallel}, --blocks {vanilla,sequential,parallel}
                        Choose the type of architecture
  -nb NUM_BLOCKS, --num_blocks NUM_BLOCKS
                        If not 'vanilla' specifiy as a second argument the number of blocks
  --split SPLIT         Train-test split percentage as an integer from 0 to 100. For batch training splits accross subjects; otherwise, accross time
                        series length.
  --skip SKIP           Number of time points to skip when testing predictability
  --length LENGTH       Length of the time series to analyse
  --subjects [SUBJECTS ...]
                        List of subjects to process. Default is all. Type -1 for all.
  --rois ROIS [ROIS ...]
                        Space separated list of ROIs to analyse. Set to -1 for whole network analysis. Default is -1
  --num_surrogates NUM_SURROGATES
                        Number of surrogates to generate
  --batch_analysis      Train the reservoirs on a batch of time series instead of single training. If not present, a different reservoir will be
                        trained for each time series and the results will be avraged.
  --runs RUNS           In the case of single subject training, number of times to train the reservoir on a specific task
```

## To analyse fmri time series
```shell
user@user:~/Repository$ python main_RCCausality.py "" fmri --help
usage: Compute Reservoir Computing Causality on time series.
In development.
 dir fmri [-h] [--deconvolve DECONVOLVE [DECONVOLVE ...]]

options:
  -h, --help            show this help message and exit
  --deconvolve DECONVOLVE [DECONVOLVE ...]
                        NOT IMPLEMENTED

```

## To analyse a logistic mapping between two variables
```shell
user@user:~/Repository$ python main_RCCausality.py "" logistic  --help
usage: Compute Reservoir Computing Causality on time series.
In development.
 dir logistic [-h] [--generate] [--num_points NUM_POINTS]
                                                                                           [--lags_x2y LAGS_X2Y [LAGS_X2Y ...]]
                                                                                           [--lags_y2x LAGS_Y2X [LAGS_Y2X ...]]
                                                                                           [--c_x2y C_X2Y [C_X2Y ...]] [--c_y2x C_Y2X [C_Y2X ...]]
                                                                                           [--samples SAMPLES] [--noise NOISE [NOISE ...]]
                                                                                           [--convolve CONVOLVE]

options:
  -h, --help            show this help message and exit
  --generate            Generate logistic time series
  --num_points NUM_POINTS
                        Number of time points to generate
  --lags_x2y LAGS_X2Y [LAGS_X2Y ...]
                        Lags where the causal relationship from x to y take place
  --lags_y2x LAGS_Y2X [LAGS_Y2X ...]
                        Lags where the causal relationship from y to x take place
  --c_x2y C_X2Y [C_X2Y ...]
                        Strengths of the causal relationship from x to y take place
  --c_y2x C_Y2X [C_Y2X ...]
                        Strengths of the causal relationship from y to x take place
  --samples SAMPLES     Number of samples to generate - they will be treated as subjects
  --noise NOISE [NOISE ...]
                        Coupling and Standard deviation of the white noise to be added
  --convolve CONVOLVE   Kernel size of the filter to convolve. If not specified no convolution will be applied.
```