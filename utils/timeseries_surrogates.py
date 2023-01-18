# This code is LITERALLY COPIED from the pyunicorn package. The issue arose due to the inability to 
#   install the whole package. Pip was throwing an error that was apparently related to the setup.py
#   script of the library itself. Anaconda was installing something that I was not sure it was the 
#   correct package. 

# As a solution, I copied the function that were relevant to our problem from the source code in the 
#   following link: https://github.com/pik-copan/pyunicorn/blob/master/src/pyunicorn/timeseries/surrogates.py

# Consider acknowledging the corresponding reference as taken from the documentation page:
#   J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng, L. Tupikina, V. Stolbova, R.V. Donner, 
#   N. Marwan, H.A. Dijkstra, and J. Kurths, Unified functional network and nonlinear time series analysis for 
#   complex systems science: The pyunicorn package, Chaos 25, 113101 (2015), doi:10.1063/1.4934554, 
#   Preprint: arxiv.org:1507.01571 [physics.data-an].

import numpy as np

def correlated_noise_surrogates(original_data):
    """
    Return Fourier surrogates.
    Generate surrogates by Fourier transforming the original_data
    time series (assumed to be real valued), randomizing the phases and
    then applying an inverse Fourier transform. Correlated noise surrogates
    share their power spectrum and autocorrelation function with the
    original_data time series.
    .. note::
        The amplitudes are not adjusted here, i.e., the
        individual amplitude distributions are not conserved!
    **Examples:**
    The power spectrum is conserved up to small numerical deviations.
    However, the time series amplitude distributions differ.
    TODO: Add arguments and returns
    
    Inputs:
    ==============
    original data: (numpy array of shape (N, n_time))

    Outputs:
    ==============
    """

    #  Calculate FFT of original_data time series
    surrogates = np.fft.rfft(original_data, axis=1)

    #  Get shapes
    N, n_time = original_data.shape
    len_phase = surrogates.shape[1]

    #  Generate random phases uniformly distributed in the
    #  interval [0, 2*Pi]
    phases = np.random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

    #  Add random phases uniformly distributed in the interval [0, 2*Pi]
    surrogates *= np.exp(1j * phases)

    #  Calculate IFFT and take the real part, the remaining imaginary part
    #  is due to numerical errors.
    return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,
                                                        axis=1)))

def AAFT_surrogates(original_data):
    """
    Return surrogates using the amplitude adjusted Fourier transform
    method.
    Reference: [Schreiber2000]_
    TODO: Add arguments and returns
    
    Inputs:
    ==============
    original data: (numpy array of shape (N, n_time))

    Outputs:
    ==============
    """

    #  Create sorted Gaussian reference series
    gaussian = np.random.randn(original_data.shape[0], original_data.shape[1])
    gaussian.sort(axis=1)

    #  Rescale data to Gaussian distribution
    ranks = original_data.argsort(axis=1).argsort(axis=1)
    rescaled_data = np.zeros(original_data.shape)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = gaussian[i, ranks[i, :]]

    #  Phase randomize rescaled data
    phase_randomized_data = correlated_noise_surrogates(rescaled_data)

    #  Rescale back to amplitude distribution of original data
    sorted_original = original_data.copy()
    sorted_original.sort(axis=1)

    ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = sorted_original[i, ranks[i, :]]

    return rescaled_data

def refined_AAFT_surrogates(original_data, n_iterations, output="true_amplitudes"):
    """
    Known as Iterative AAFT (IAAFT) in Lucio, et al. Phys. Rev. E. (2012).
    Return surrogates using the iteratively refined amplitude adjusted
    Fourier transform method. 
    A set of AAFT surrogates (:meth:`AAFT_surrogates`) is iteratively
    refined to produce a closer match of both amplitude distribution and
    power spectrum of surrogate and original data.
    Reference: [Schreiber2000]_
    TODO: Add arguments and returns
    
    Inputs:
    ==============
    original data: (numpy array of shape (N, n_time))

    Outputs:
    ==============
    """

    #  Get size of dimensions
    n_time = original_data.shape[1]

    #  Get Fourier transform of original data with caching
    fourier_transform = np.fft.rfft(original_data, axis=1)

    #  Get Fourier amplitudes
    original_fourier_amps = np.abs(fourier_transform)

    #  Get sorted copy of original data
    sorted_original = original_data.copy()
    sorted_original.sort(axis=1)

    #  Get starting point / initial conditions for R surrogates
    # (see [Schreiber2000]_)
    R = AAFT_surrogates(original_data)

    #  Start iteration
    for i in range(n_iterations):
        #  Get Fourier phases of R surrogate
        r_fft = np.fft.rfft(R, axis=1)
        r_phases = r_fft / np.abs(r_fft)

        #  Transform back, replacing the actual amplitudes by the desired
        #  ones, but keeping the phases exp(iψ(i)
        s = np.fft.irfft(original_fourier_amps * r_phases, n=n_time,
                            axis=1)

        #  Rescale to desired amplitude distribution
        ranks = s.argsort(axis=1).argsort(axis=1)

        for j in range(original_data.shape[0]):
            R[j, :] = sorted_original[j, ranks[j, :]]

    if output == "true_amplitudes":
        return R
    elif output == "true_spectrum":
        return s
    elif output == "both":
        return (R, s)
    else:
        return (R, s) 

if __name__ == '__main__':
    pass