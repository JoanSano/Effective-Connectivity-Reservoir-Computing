import argparse
from datetime import datetime
import json

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
    fmri.add_argument('--dir', type=str, default='./Datasets/HCP_motor-task_12-subjects', help="Relative path pointing to the directory where the data is stored")
    fmri.add_argument('--subjects', type=str, default=['-1'], nargs='*', help="List of subjects to process. Default is all. Type -1 for all.")
    fmri.add_argument('--rois', type=int, default=[-1], nargs='+', help="Space separated list of ROIs to analyse. Set to -1 for whole brain analysis. Default is -1")
    fmri.add_argument('--split', type=int, default=75, help="Train-test split percentage as an integer from 0 to 100")
    fmri.add_argument('--skip', type=int, default=20, help="Number of time points to skip when testing predictability")
    fmri.add_argument('--runs', type=int, default=1, help="Number of times to run RCC on a given pair of real samples")

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

    logistic = sub_parser.add_parser('logistic', help="Ignore")
    logistic.add_argument('-i', '--ignore', default=None, help="Ignore. this branch will include several features for the logistic time series")

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

    ###########################################
    # Positional arguments (mutually exclusive) regarding which time series to analyse
    timeseries = parser.add_subparsers()    
    timeseries = fmri_arguments(timeseries) # fMRI    
    timeseries = logistic_arguments(timeseries) # Logistic 

    # Optional arguments -- Reservoir architecture and parameters
    parser = optional_arguments(parser)    

    # Parse arguiments and extract the timeseries present in command line
    opts = parser.parse_args()
    try:
        timeseries_type = opts.func() 
        return opts, timeseries_type
    except:
        print("InputError: Missing positional argument specifying the time series; choose from: {fmri,logistic}")
        quit()
    
if __name__ == '__main__':
    pass