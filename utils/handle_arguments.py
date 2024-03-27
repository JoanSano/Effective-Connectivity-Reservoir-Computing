import argparse
from datetime import datetime
import os
import json

class NameSpace(object):
    pass

def add_json_data_to_parser(data):
    # Creating instance of NameSpace:
    opts = NameSpace()

    # Checking inputs
    allowed = [
        "dir", "r_folder", "num_jobs", "length", "subjects", "rois", 
            "min_lag", "max_lag", "blocks", "num_blocks", "split", "skip",
            "num_surrogates", "runs",
        "fmri", "deconvolve",
        "logistic", "generate", "num_points", "lags_x2y", "lags_y2x",
            "c_x2y", "c_y2x", "samples", "noise", "convolve"
    ]
    exclusive = ["fmri", "logistic"]
    k = 0
    for a in data.keys():
        k = k+1 if a in exclusive else k
    if k!=1:
        raise NameError(f"Please provide only one of the following: {exclusive}")
    
    # Generating object
    for key, value in data.items():
        if key not in allowed:
            raise NameError(f"Unrecognized argument: {key}")
        else:
            #parser.add_argument(f'--{key}', type=type(value), default=value)
            setattr(opts, key, value)
    return opts

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

    # Arguments from command line
    if isinstance(main_parser, argparse.ArgumentParser):
        def_folder = 'Results' + datetime.now().strftime("%d-%m-%Y_%H-%M")
        main_parser.add_argument('-rf','--r_folder', type=str, default=def_folder, help="Output directory where results will be stored")
        main_parser.add_argument('-j', '--num_jobs', type=int, default=2, help='Number of parallel jobs to launch')
        main_parser.add_argument('-b', '--blocks', type=str, choices=['vanilla', 'sequential', 'parallel'], default="vanilla", help="Choose the type of architecture")
        main_parser.add_argument('-nb', '--num_blocks', type=int, default=None, help="If not 'vanilla' specifiy as a second argument the number of blocks")
        main_parser.add_argument('--split', type=int, default=80, help="Train split percentage (int) from 0 to 100 (0 and 100 means no split). For batch training splits accross subjects; otherwise, accross time series length; k-fold CV specified by -k")
        main_parser.add_argument('--skip', type=int, default=10, help="Number of time points to skip when testing predictability")
        main_parser.add_argument('--length', type=int, default=100, help="Percentage of the length of the time series to analyse")
        main_parser.add_argument('--subjects', type=str, default=['-1'], nargs='*', help="List of subjects to process. Default is all. Type -1 for all.")
        main_parser.add_argument('--rois', type=int, default=[-1], nargs='+', help="Space separated list of ROIs to analyse. Set to -1 for whole network analysis. Default is -1")
        main_parser.add_argument('--num_surrogates', type=int, default=100, help="Number of surrogates to generate")
        main_parser.add_argument('--min_lag', type=int, default=-30, help="Minimum value of the negative lag to test")
        main_parser.add_argument('--max_lag', type=int, default=31, help="Maximum value of the positive lag to test")
        main_parser.add_argument('--runs', type=int, default=5, help="Number of times to train the reservoir with the real samples")

    # Arguments from json file or dict
    else: 
        present_args = data.keys()
        if "r_folder" not in present_args:
            def_folder = 'Results' + datetime.now().strftime("%d-%m-%Y_%H-%M")
            setattr(main_parser, "r_folder", def_folder)
        if "num_jobs" not in present_args:
            setattr(main_parser, "num_jobs", 2)
        if "blocks" not in present_args:
            setattr(main_parser, "blocks", "vanilla")
        if "num_blocks" not in present_args:
            setattr(main_parser, "num_blocks", None)
        if "split" not in present_args:
            setattr(main_parser, "split", 80)
        if "skip" not in present_args:
            setattr(main_parser, "skip", 10)
        if "length" not in present_args:
            setattr(main_parser, "length", 100)
        if "subjects" not in present_args:
            setattr(main_parser, "subjects", '-1')
        if "rois" not in present_args:
            setattr(main_parser, "rois", [-1])
        if "num_surrogates" not in present_args:
            setattr(main_parser, "num_surrogates", 100)
        if "min_lag" not in present_args:
            setattr(main_parser, "min_lag", -30)
        if "max_lag" not in present_args:
            setattr(main_parser, "max_lag", 31)
        if "runs" not in present_args:
            setattr(main_parser, "runs", 5)
    
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

    # Arguments from command line
    if data is None:
        fmri = sub_parser.add_parser('fmri', help="Analyse fMRI time series; Use the flag [(-h,--help) HELP] to see optional inputs")
        fmri.add_argument('--deconvolve', type=int, default=None, nargs='+', help="NOT IMPLEMENTED")
        # fmri positional argument is present
        fmri.set_defaults(func=lambda: 'fmri') 

    # Arguments from json file or dict
    else: 
        present_args = data.keys()
        if "fmri" in present_args and data["fmri"]:
            if "deconvolve" not in present_args:
                setattr(sub_parser, "deconvolve", None)

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

    # Arguments from command line
    if data is None:
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

    # Arguments from json file or dict
    else: 
        present_args = data.keys()
        if "logistic" in present_args and data["logistic"]:
            if "generate" in present_args and data["generate"]:
                if "num_points" not in present_args: 
                    setattr(sub_parser, "num_points", 250)
                if "lags_x2y" not in present_args:    
                    setattr(sub_parser, "lags_x2y", [2])
                if "lags_y2x" not in present_args:    
                    setattr(sub_parser, "lags_y2x", None)
                if "c_x2y" not in present_args:    
                    setattr(sub_parser, "c_x2y", [0.8])
                if "c_y2x" not in present_args:    
                    setattr(sub_parser, "c_y2x", None)
                if "samples" not in present_args:    
                    setattr(sub_parser, "samples", 10)
                if "noise" not in present_args:    
                    setattr(sub_parser, "noise", [0, 1])
                if "convolve" not in present_args:    
                    setattr(sub_parser, "convolve", None)
    
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
    # Arguments passed through command line
    if args is None: 
        parser = argparse.ArgumentParser("\nCompute Reservoir Computing Causality on time series.\nIn development.\n")
        parser.add_argument('dir', type=str, default='./Datasets/Unknown', help="Relative path pointing to the directory where the data is stored and/or generated")

        # Optional arguments -- Reservoir architecture and parameters
        parser = optional_arguments(parser)
        
        # Positional arguments (mutually exclusive) regarding which time series to analyse
        timeseries = parser.add_subparsers()    
        timeseries = fmri_arguments(timeseries) # fMRI    
        timeseries = logistic_arguments(timeseries) # Logistic 

        # Parse arguiments and extract the timeseries present in command line
        opts, _ = parser.parse_known_args()
    
    # ============================================ #
    # Arguments passed as a json file (i.e., string)
    elif isinstance(args, str):     
        with open(args, 'rt') as f:
            args_keys = json.load(f)
        # Main arguments
        opts = add_json_data_to_parser(args_keys)  
        # Optional arguments
        opts = optional_arguments(opts, data=args_keys) 
        # Exclusive arguments
        opts = fmri_arguments(opts, data=args_keys) # fMRI    
        opts = logistic_arguments(opts, data=args_keys) # Logistic  

    # ============================================ # 
    # Arguments passed as a json file (i.e., string)
    elif isinstance(args, dict): 
        args_keys = args
        # Main arguments
        opts = add_json_data_to_parser(args_keys)  
        # Optional arguments
        opts = optional_arguments(opts, data=args_keys) 
        # Exclusive arguments
        opts = fmri_arguments(opts, data=args_keys) # fMRI    
        opts = logistic_arguments(opts, data=args_keys) # Logistic 
    else: 
        raise TypeError("Please provide input arguments in the form of --flags -F (i.e., command line), json file or python dictionary")
    
    # Get time series type
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
    # Making path as absolutes
    opts.dir = os.path.abspath(opts.dir)
    opts.r_folder = os.path.abspath(opts.r_folder)
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
        try:
            os.mkdir(results_dir)
        except:
            # The os control flag did not work due to simultaneous jobs running
            pass

    # Corresponding data files
    if timeseries_type == 'logistic':
        if hasattr(opts, 'generate') and opts.generate:        
            from utils.generate_logistic import generate_series
            generate_series(opts)
    files = [os.path.join(opts.dir, f) for f in os.listdir(opts.dir) if f.split(".")[0] in opts.subjects or opts.subjects[0] == '-1']
    
    # Reservoir Architecture parameters file
    json_config = './reservoir_config.json'
    if not os.path.exists(results_dir+json_config):
        os.system(f"cp {json_config} {results_dir}")
    else:
        os.system(f"rm {results_dir}/reservoir_config.json")
        os.system(f"cp {json_config} {results_dir}")

    # Drop command line call for transparency
    with open(os.path.join(results_dir, 'commandline_args.txt'), 'w') as f:
        for arg, val in zip(opts.__dict__.keys(),opts.__dict__.values()):
            f.write(arg+': '+str(val)+'\n')

    return opts, files, results_dir, json_config, timeseries_type

if __name__ == '__main__':
    pass