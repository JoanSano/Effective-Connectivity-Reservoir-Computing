## Relative imports
from execution.utils import run_RCC, run_GC

if __name__ == '__main__':
    method = "RCC"
    if method == "RCC":
        run_RCC()   
    elif method == "GC":
        run_GC()   
    elif method == "CCS":
        raise NotImplementedError
    else:
        raise NotImplementedError