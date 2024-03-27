import glob
import os

# Relative imports
from analysis.ground_truths.utils import create_logistic_networks

class Logistic():
    def __init__(self, cmd_dir: str, x2y: list=None, y2x: list=None) -> None:
        # Load arguments
        self.cmd_dir = os.path.abspath(cmd_dir)
        with open(os.path.join(self.cmd_dir,"commandline_args.txt")) as args:
            for arg in args.readlines():
                (key,val) = arg.strip().split(": ")
                setattr(Logistic, key, val)

        # Process attributes (slightly)
        self.lags_x2y = None if self.lags_x2y=='None' else self.lags_x2y[1:-1].split(",")
        self.lags_y2x = None if self.lags_y2x=='None' else self.lags_y2x[1:-1].split(",")
        self.c_x2y = None if self.c_x2y=='None' else self.c_x2y[1:-1].split(",")
        self.c_y2x = None if self.c_y2x=='None' else self.c_y2x[1:-1].split(",")

        self.lags = range(int(self.min_lag),int(self.max_lag))
        self.x2y, self.y2x = {}, {}
        if self.lags_x2y is not None:
            for l, c in zip(self.lags_x2y,self.c_x2y):
                self.x2y[-int(l)] = float(c)
                if int(l)<int(self.max_lag):
                    self.x2y[int(l)] = float(c) 
        if self.lags_y2x is not None:
            for l, c in zip(self.lags_y2x,self.c_y2x):
                self.y2x[-int(l)] = float(c)
                if int(l)<int(self.max_lag):
                    self.y2x[int(l)] = float(c) 
        
    # Create networks
    def networks(self):
        return create_logistic_networks(self.lags, self.x2y, self.y2x)