import glob
import os

# Relative imports
from analysis.loaders import Group_EC
from analysis.ground_truths.utils import load_netsim_networks
from analysis.utils import Constraints

class Netsim(Group_EC, Constraints):
    def __init__(self, results_dir: str, simulation: int=None, ROI_Labels=None) -> None:
        # Initialize the EC reader
        Group_EC.__init__(self, directory=results_dir, ROI_Labels=ROI_Labels)

        # Load arguments from the command line input used to run RCC
        with open(os.path.join(os.path.abspath(self.directory),"commandline_args.txt")) as args:
            for arg in args.readlines():
                (key,val) = arg.strip().split(": ")
                setattr(Netsim, key, val)

        # Load the ground truths from the corresponding Dataset folder
        self.dir = "/".join(self.dir.split("/")[:-1])        
        self.simulation = f"sim-{simulation}" #self.dir.split("/")[-2] 
        self.weighted_gts, self.binary_gt = load_netsim_networks(
            self.dir, self.simulation, [s.split("_")[0] for s in self.subject_list]
        )
        
        # Initialize the methods from Constraints
        Constraints.__init__(self, structure=self.binary_gt)