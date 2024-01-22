import argparse
from datetime import datetime
import os
import json

def add_json_data_to_parser(parser, data):
    for key, value in data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    return parser

def optional_arguments(main_parser, data=None):
    """
    Adds optional arguments to the main argument parser of the program

    Arguments
    ---------
    main_parser: (parser object) Containing the arguments
    data: (dict) Optional dictionary containing arguments from json file or python dict -- Necessary for jupyter notebook usage

    Output
    ------
    main_parser: (object) Includes the optional command line arguments stored as attributes
    """

    # Arguments from json file or dict
    if data is not None: 
        present_args = data.keys()
        if "r_folder" not in present_args:
            def_folder = 'Results' + datetime.now().strftime("%d-%m-%Y_%H-%M")
            main_parser.add_argument('-rf','--r_folder', type=str, default=def_folder, help="Output directory where results will be stored")
        if "num_jobs" not in present_args:
            main_parser.add_argument('-j', '--num_jobs', type=int, default=2, help='Number of parallel jobs to launch')
        if "blocks" not in present_args:
            main_parser.add_argument('-b', '--blocks', type=str, choices=['vanilla', 'sequential', 'parallel'], default="vanilla", help="Choose the type of architecture")
        if "num_blocks" not in present_args:
            main_parser.add_argument('-nb', '--num_blocks', type=int, default=None, help="If not 'vanilla' specifiy as a second argument the number of blocks")
        if "split" not in present_args:
            main_parser.add_argument('--split', type=int, default=100, help="Train split percentage (int) from 0 to 100 (0 and 100 means no split). For batch training splits accross subjects; otherwise, accross time series length; k-fold CV specified by -k")
        if "skip" not in present_args:
            main_parser.add_argument('--skip', type=int, default=10, help="Number of time points to skip when testing predictability")
        if "length" not in present_args:
            main_parser.add_argument('--length', type=int, default=100, help="Length of the time series to analyse")
        if "subjects" not in present_args:
            main_parser.add_argument('--subjects', type=str, default=['-1'], nargs='*', help="List of subjects to process. Default is all. Type -1 for all.")
        if "rois" not in present_args:
            main_parser.add_argument('--rois', type=int, default=[-1], nargs='+', help="Space separated list of ROIs to analyse. Set to -1 for whole network analysis. Default is -1")
        if "num_surrogates" not in present_args:
            main_parser.add_argument('--num_surrogates', type=int, default=100, help="Number of surrogates to generate")
        if "min_lag" not in present_args:
            main_parser.add_argument('--min_lag', type=int, default=-30, help="Minimum value of the negative lag to test")
        if "max_lag" not in present_args:
            main_parser.add_argument('--max_lag', type=int, default=31, help="Maximum value of the positive lag to test")
        if "plot" not in present_args:
            main_parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help="Plot predictability scores for each pair of ROIs")
        if "runs" not in present_args:
            main_parser.add_argument('--runs', type=int, default=5, help="Number of times to train the reservoir with the real samples")
    # Arguments from command line
    else:
        def_folder = 'Results' + datetime.now().strftime("%d-%m-%Y_%H-%M")
        main_parser.add_argument('-rf','--r_folder', type=str, default=def_folder, help="Output directory where results will be stored")
        main_parser.add_argument('-j', '--num_jobs', type=int, default=2, help='Number of parallel jobs to launch')
        main_parser.add_argument('-b', '--blocks', type=str, choices=['vanilla', 'sequential', 'parallel'], default="vanilla", help="Choose the type of architecture")
        main_parser.add_argument('-nb', '--num_blocks', type=int, default=None, help="If not 'vanilla' specifiy as a second argument the number of blocks")
        main_parser.add_argument('--split', type=int, default=100, help="Train split percentage (int) from 0 to 100 (0 and 100 means no split). For batch training splits accross subjects; otherwise, accross time series length; k-fold CV specified by -k")
        main_parser.add_argument('--skip', type=int, default=10, help="Number of time points to skip when testing predictability")
        main_parser.add_argument('--length', type=int, default=100, help="Length of the time series to analyse")
        main_parser.add_argument('--subjects', type=str, default=['-1'], nargs='*', help="List of subjects to process. Default is all. Type -1 for all.")
        main_parser.add_argument('--rois', type=int, default=[-1], nargs='+', help="Space separated list of ROIs to analyse. Set to -1 for whole network analysis. Default is -1")
        main_parser.add_argument('--num_surrogates', type=int, default=100, help="Number of surrogates to generate")
        main_parser.add_argument('--min_lag', type=int, default=-30, help="Minimum value of the negative lag to test")
        main_parser.add_argument('--max_lag', type=int, default=31, help="Maximum value of the positive lag to test")
        main_parser.add_argument('--plot', action=argparse.BooleanOptionalAction, help="Plot predictability scores for each pair of ROIs")
        main_parser.add_argument('--runs', type=int, default=5, help="Number of times to train the reservoir with the real samples")

    return main_parser

def fmri_arguments(sub_parser, data=None):
    """
    Adds an additional fmri argument parser as well as its options

    Arguments
    ---------
    sub_parser: (parser object) Subparser object
    data: (dict) Optional dictionary containing arguments from json file or python dict -- Necessary for jupyter notebook usage

    Output
    ------
    sub_parser: (object) Includes the optional command line arguments associated to the fmri parser stored as attributes
    """

    # Arguments from json file or dict
    if data is not None:
        present_args = data.keys()
        if "fmri" in present_args and data["fmri"]:
            fmri = sub_parser.add_parser('fmri', help="Analyse fMRI time series; Use the flag [(-h,--help) HELP] to see optional inputs")
            if "deconvolve" not in present_args:
                fmri.add_argument('--deconvolve', type=int, default=[-1], nargs='+', help="NOT IMPLEMENTED")
    # Arguments from command line
    else:
        fmri = sub_parser.add_parser('fmri', help="Analyse fMRI time series; Use the flag [(-h,--help) HELP] to see optional inputs")
        fmri.add_argument('--deconvolve', type=int, default=[-1], nargs='+', help="NOT IMPLEMENTED")
        # fmri positional argument is present
        fmri.set_defaults(func=lambda: 'fmri') 

    return sub_parser

def logistic_arguments(sub_parser, data=None):
    """
    Adds optional arguments to the logistic argument parser

    Arguments
    ---------
    sub_parser: (parser object) Subparser object
    data: (dict) Optional dictionary containing arguments from json file or python dict -- Necessary for jupyter notebook usage

    Output
    ------
    sub_parser: (object) Includes the optional command line arguments associated to the fmri parser stored as attributes
    """

    # Arguments from json file or dict
    if data is not None:
        present_args = data.keys()
        if "logistic" in present_args and data["logistic"]:
            logistic = sub_parser.add_parser('logistic', help="Anlysis of logistic time series to test the method; Use the flag [(-h,--help) HELP] to see optional inputs")
            if "generate" in present_args and data["generate"]:
                logistic.add_argument('--generate', action='store_true', help="Generate logistic time series")
                if "num_points" in present_args: 
                    logistic.add_argument('--num_points', type=int, default=250, help="Number of time points to generate")
                if "lags_x2y" in present_args:    
                    logistic.add_argument('--lags_x2y', type=int, default=[2], nargs='+', help="Lags where the causal relationship from x to y take place")
                if "lags_y2x" in present_args:    
                    logistic.add_argument('--lags_y2x', type=int, default=None, nargs='+', help="Lags where the causal relationship from y to x take place")
                if "c_x2y" in present_args:    
                    logistic.add_argument('--c_x2y', type=float, default=[0.8], nargs='+', help="Strengths of the causal relationship from x to y take place")
                if "genec_y2xrate" in present_args:    
                    logistic.add_argument('--c_y2x', type=float, default=None, nargs='+', help="Strengths of the causal relationship from y to x take place")
                if "samples" in present_args:    
                    logistic.add_argument('--samples', type=int, default=10, help="Number of samples to generate - they will be treated as subjects")
                if "noise" in present_args:    
                    logistic.add_argument('--noise', type=float, default=[0, 1], nargs='+', help="Coupling and Standard deviation of the white noise to be added")
                if "convolve" in present_args:    
                    logistic.add_argument('--convolve', type=int, default=None, help="Kernel size of the filter to convolve. If not specified no convolution will be applied.")
    # Arguments from command line
    else:
        logistic = sub_parser.add_parser('logistic', help="Anlysis of logistic time series to test the method; Use the flag [(-h,--help) HELP] to see optional inputs")
        logistic.add_argument('--generate', action='store_true', help="Generate logistic time series")
        logistic.add_argument('--num_points', type=int, default=250, help="Number of time points to generate")
        logistic.add_argument('--lags_x2y', type=int, default=[2], nargs='+', help="Lags where the causal relationship from x to y take place")
        logistic.add_argument('--lags_y2x', type=int, default=None, nargs='+', help="Lags where the causal relationship from y to x take place")
        logistic.add_argument('--c_x2y', type=float, default=[0.8], nargs='+', help="Strengths of the causal relationship from x to y take place")
        logistic.add_argument('--c_y2x', type=float, default=None, nargs='+', help="Strengths of the causal relationship from y to x take place")
        logistic.add_argument('--samples', type=int, default=10, help="Number of samples to generate - they will be treated as subjects")
        logistic.add_argument('--noise', type=float, default=[0, 1], nargs='+', help="Coupling and Standard deviation of the white noise to be added")
        logistic.add_argument('--convolve', type=int, default=None, help="Kernel size of the filter to convolve. If not specified no convolution will be applied.")
        # logistic positional argument is present
        logistic.set_defaults(func=lambda: 'logistic')

    return sub_parser

def handle_argumrnts(args=None): 
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
    if args is None: # Arguments passed through command line
        args_keys = None
        parser.add_argument('dir', type=str, default='./Datasets/Logistic', help="Relative path pointing to the directory where the data is stored and/or generated")
    elif isinstance(args, str): # Arguments passed as a json file (i.e., string)    
        with open(args, 'rt') as f:
            args_keys = json.load(f)
            parser = add_json_data_to_parser(parser, args_keys)            
    elif isinstance(args, dict): # Arguments passed as a json file (i.e., string)
        args_keys = args
        parser = add_json_data_to_parser(parser, args_keys)   
    else: 
        raise TypeError("Please provide input arguments in the form of --flags -F (i.e., command line), json file or python dictionary")
    
    # Optional arguments -- Reservoir architecture and parameters
    parser = optional_arguments(parser, data=args_keys)
    
    ###########################################
    # Positional arguments (mutually exclusive) regarding which time series to analyse
    timeseries = parser.add_subparsers()    
    timeseries = fmri_arguments(timeseries, data=args_keys) # fMRI    
    timeseries = logistic_arguments(timeseries, data=args_keys) # Logistic 
    
    # Parse arguiments and extract the timeseries present in command line
    #opts = parser.parse_args()
    opts, _ = parser.parse_known_args()
    print(opts)
    if hasattr(opts, 'fmri'):
        timeseries_type = 'fmri'
    elif hasattr(opts, 'logistic'):
        timeseries_type = 'logistic'
    else:
        try:
            timeseries_type = opts.func() 
        except:
            print("InputError: Missing positional argument specifying the time series; choose from: {fmri,logistic,...}")
            quit()

    return opts, timeseries_type

def initialize_and_grep_files(args=None):
    """
    TODO: Add documentation
    """
    
    # Execution options
    opts, timeseries_type = handle_argumrnts(args=args)
    
    # Creating necessary paths
    root_dir = os.getcwd()
    results_dir = os.path.join(root_dir, opts.r_folder)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    # Corresponding data files
    if timeseries_type == 'logistic' and hasattr(opts, 'generate'):        
        from utils.generate_logistic import generate_series
        generate_series(opts)
    files = [os.path.join(opts.dir, f) for f in os.listdir(opts.dir) if f.split(".")[0] in opts.subjects or opts.subjects[0] == '-1']
    
    # Reservoir Architecture parameters file
    json_config = './reservoir_config.json'
    os.system(f"cp {json_config} {results_dir}")

    # Drop command line call for transparency
    with open(os.path.join(results_dir, 'commandline_args.txt'), 'w') as f:
        for arg, val in zip(opts.__dict__.keys(),opts.__dict__.values()):
            f.write(arg+': '+str(val)+'\n')
            
    return opts, files, results_dir, json_config, timeseries_type

if __name__ == '__main__':
    pass