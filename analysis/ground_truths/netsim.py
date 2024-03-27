import glob
import os

# Relative imports
from analysis.ground_truths.utils import load_netsim_networks
from analysis.utils import Constraints

class Netsim(Constraints):
    def __init__(self, cmd_dir: str, subjects: list=None) -> None:

        # Load arguments
        self.cmd_dir = os.path.abspath(cmd_dir)
        with open(os.path.join(self.cmd_dir,"commandline_args.txt")) as args:
            for arg in args.readlines():
                (key,val) = arg.strip().split(": ")
                setattr(Netsim, key, val)
        self.simulation = self.dir.split("/")[-2] 
        if subjects is None:
            self.subjects = [s.split("/")[-2].split("_")[0] for s in glob.glob(self.cmd_dir+"/*/")]
        else:
            self.subjects = subjects
        self.dir = "/".join(self.dir.split("/")[:-1])
        
        self.weighted_gts, self.binary_gt = load_netsim_networks(self.dir, self.simulation, self.subjects)
        super().__init__(self.binary_gt)