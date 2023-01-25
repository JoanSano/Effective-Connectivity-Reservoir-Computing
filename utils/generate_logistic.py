import numpy as np
import os

def generate_series(opts):    
    """
    TODO: Add documentation
    """

    dir = os.path.join(os.getcwd(), opts.dir)
    if not os.path.exists(dir):
        os.mkdir(dir)

    for sample in range(1, opts.samples+1):
        x, y = [], []
        x0, y0 = 0.1, 0.2
        x.append(x0), y.append(y0)

        for i in range(1, opts.num_points):
            xi = x[i-1]*(3.78-3.78*x[i-1]) 
            yi = y[i-1]*(3.77-3.77*y[i-1])

            # y --> x
            if opts.lags_y2x and opts.c_y2x:
                for lag, coupling in zip(opts.lags_y2x, opts.c_y2x):
                    if i >= lag:
                        xi += -coupling * x[i-1] * y[i-lag]

            if opts.lags_x2y and opts.c_x2y:
                # x --> y
                for lag, coupling in zip(opts.lags_x2y, opts.c_x2y):
                    if i >= lag:
                        yi += -coupling * y[i-1] * x[i-lag]
            xi, yi = round(xi, 10), round(yi,10)
            x.append(xi), y.append(yi)
            
        # TODO: Add convolution to hide causal relationships

        to_save = np.zeros((opts.num_points,3))
        to_save[:,0] = np.array(x) * np.nan
        to_save[:,1] = np.array(x) + opts.noise[0] * np.random.normal(0, opts.noise[1])
        to_save[:,2] = np.array(y) + opts.noise[0] * np.random.normal(0, opts.noise[1])

        name = os.path.join(dir, "sub-"+str(sample)+"_logistic_TS.txt")
        np.savetxt(name, to_save, delimiter='\t')    

if __name__=='__main__':
    pass