import glob
import os

# Relative imports
from analysis.loaders import Group_EC
from analysis.ground_truths.utils import load_var_network
from analysis.utils import Constraints

class var(Group_EC, Constraints):
    def __init__(self, results_dir: str, ROI_Labels=None) -> None:
        # Initialize the EC reader
        Group_EC.__init__(self, directory=results_dir, ROI_Labels=ROI_Labels)

        # Load arguments from the command line input used to run the estimator
        with open(os.path.join(os.path.abspath(self.directory),"commandline_args.txt")) as args:
            for arg in args.readlines():
                (key,val) = arg.strip().split(": ")
                setattr(var, key, val)

        # Load the ground truths from the corresponding Dataset folder
        self.dir = "/".join(self.dir.split("/")[:-1])        
        self.weighted_gt, self.binary_gt = load_var_network(self.dir)
        
        # Initialize the methods from Constraints
        Constraints.__init__(self, structure=self.binary_gt)