import argparse
from datetime import datetime
import os

def optional_arguments(main_parser):
    """
    Adds optional arguments to the main argument parser of the program

    Arguments
    ---------
    main_parser: (parser object) Containing the arguments

    Output
    ------
    main_parser: (object) Includes the optional command line arguments stored as attributes
    """
     
    def_folder = 'Results' + datetime.now().strftime("%d-%m-%Y_%H-%M")
    main_parser.add_argument('-rf','--r_folder', type=str, default=def_folder, help="Output directory where results will be stored")
    main_parser.add_argument('-j', '--num_jobs', type=int, default=2, help='Number of parallel jobs to launch')
    main_parser.add_argument('-b', '--blocks', type=str, choices=['vanilla', 'sequential', 'parallel'], default="vanilla", help="Choose the type of architecture")
    main_parser.add_argument('-nb', '--num_blocks', type=int, default=None, help="If not 'vanilla' specifiy as a second argument the number of blocks")
    main_parser.add_argument('--split', type=int, default=100, help="Train-test split percentage as an integer from 0 to 100. For batch training splits accross subjects; otherwise, accross time series length.")
    main_parser.add_argument('--skip', type=int, default=10, help="Number of time points to skip when testing predictability")
    main_parser.add_argument('--length', type=int, default=100, help="Length of the time series to analyse")
    main_parser.add_argument('--subjects', type=str, default=['-1'], nargs='*', help="List of subjects to process. Default is all. Type -1 for all.")
    main_parser.add_argument('--rois', type=int, default=[-1], nargs='+', help="Space separated list of ROIs to analyse. Set to -1 for whole network analysis. Default is -1")
    main_parser.add_argument('--num_surrogates', type=int, default=100, help="Number of surrogates to generate")
   
    group = main_parser.add_mutually_exclusive_group()
    group.add_argument('--batch_analysis', action='store_true', help="Train the reservoirs on a batch of time series instead of single training. If not present, a different reservoir will be trained for each time series and the results will be avraged.")
    group.add_argument('--runs', type=int, default=5, help="In the case of single subject training, number of times to train the reservoir on a specific task")

    return main_parser

def fmri_arguments(sub_parser):
    """
    Adds an additional fmri argument parser as well as its options

    Arguments
    ---------
    sub_parser: (parser object) Subparser object

    Output
    ------
    sub_parser: (object) Includes the optional command line arguments associated to the fmri parser stored as attributes
    """

    fmri = sub_parser.add_parser('fmri', help="Analyse fMRI time series; Use the flag [(-h,--help) HELP] to see optional inputs")
    fmri.add_argument('--deconvolve', type=int, default=[-1], nargs='+', help="NOT IMPLEMENTED")
    # fmri positional argument is present
    fmri.set_defaults(func=lambda: 'fmri') 

    return sub_parser

def logistic_arguments(sub_parser):
    """
    Adds optional arguments to the logistic argument parser

    Arguments
    ---------
    sub_parser: (parser object) Subparser object

    Output
    ------
    sub_parser: (object) Includes the optional command line arguments associated to the fmri parser stored as attributes
    """

    logistic = sub_parser.add_parser('logistic', help="Anlysis of logistic time series to test the method; Use the flag [(-h,--help) HELP] to see optional inputs")
    logistic.add_argument('--generate', action='store_true', help="Generate logistic time series")
    logistic.add_argument('--num_points', type=int, default=250, help="Number of time points to generate")
    logistic.add_argument('--lags_x2y', type=int, default=[2], nargs='+', help="Lags where the causal relationship from x to y take place")
    logistic.add_argument('--lags_y2x', type=int, default=None, nargs='+', help="Lags where the causal relationship from y to x take place")
    logistic.add_argument('--c_x2y', type=int, default=[0.8], nargs='+', help="Strengths of the causal relationship from x to y take place")
    logistic.add_argument('--c_y2x', type=int, default=None, nargs='+', help="Strengths of the causal relationship from y to x take place")
    logistic.add_argument('--samples', type=int, default=10, help="Number of samples to generate - they will be treated as subjects")
    logistic.add_argument('--noise', type=float, default=[0, 1], nargs='+', help="Coupling and Standard deviation of the white noise to be added")
    logistic.add_argument('--convolve', type=int, default=None, help="Kernel size of the filter to convolve. If not specified no convolution will be applied.")

    # logistic positional argument is present
    logistic.set_defaults(func=lambda: 'logistic')

    return sub_parser

def handle_argumrnts(): 
    """
    Creates the parser object that handles the command line inputs. 

    Arguments
    ---------
    None

    Output
    ------
    opts: (object) Contains the command line arguments stored as attributes
    timeseries_type: (string) Defines the type of timeseries to analyse according to what was provided in the command line

    Additional behaviors: The function terminates the program if one positional argument has not been provided
    """

    parser = argparse.ArgumentParser("\nCompute Reservoir Computing Causality on time series.\nIn development.\n")
    parser.add_argument('dir', type=str, default='./Datasets/HCP_motor-task_12-subjects', help="Relative path pointing to the directory where the data is stored and/or generated")

    # Optional arguments -- Reservoir architecture and parameters
    parser = optional_arguments(parser)

    ###########################################
    # Positional arguments (mutually exclusive) regarding which time series to analyse
    timeseries = parser.add_subparsers()    
    timeseries = fmri_arguments(timeseries) # fMRI    
    timeseries = logistic_arguments(timeseries) # Logistic 

    # Parse arguiments and extract the timeseries present in command line
    opts = parser.parse_args()
    try:
        timeseries_type = opts.func() 
        return opts, timeseries_type
    except:
        print("InputError: Missing positional argument specifying the time series; choose from: {fmri,logistic,...}")
        quit()
    
def initialize_and_grep_files():
    """
    TODO: Add documentation
    """
    
    # Execution options
    opts, timeseries_type = handle_argumrnts()
    
    # Creating necessary paths
    root_dir = os.getcwd()
    results_dir = os.path.join(root_dir, opts.r_folder)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
        
    # Drop command line call for transparency
    with open(os.path.join(results_dir, 'commandline_args.txt'), 'w') as f:
        for arg, val in zip(opts.__dict__.keys(),opts.__dict__.values()):
            f.write(arg+': '+str(val)+'\n')

    # Reservoir Architecture parameters file
    json_config = './reservoir_config.json'
    os.system(f"cp {json_config} {results_dir}")

    # Corresponding data files
    if timeseries_type == 'logistic' and opts. generate:        
        from utils.generate_logistic import generate_series
        generate_series(opts)
    files = [os.path.join(opts.dir, f) for f in os.listdir(opts.dir) if f.split(".")[0] in opts.subjects or opts.subjects[0] == '-1']

    return opts, files, results_dir, json_config, timeseries_type

if __name__ == '__main__':
    pass