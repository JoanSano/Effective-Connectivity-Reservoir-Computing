import numpy as np
from numpy.linalg import eig
import pandas as pd
from scipy.stats import ttest_ind, chi2
from scipy.integrate import cumulative_simpson
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

from utils.training_utils import input_output_lagged, split_train_test_reshape, prepare_data
from methods.reservoir_networks import reservoir_network

def reservoir_input2output(input, output, lag, I2N, N2N, split=75, skip=20, shuffle=False, axis=0, runs=None):
    """
    TODO: Add description
    """
    # Rolling the time series
    input_data, output_data = input_output_lagged(input, output, lag, axis=axis)
    
    # Split data in train and test
    input_train, output_train, input_test, output_test = split_train_test_reshape(input_data, output_data, split, shuffle=shuffle, axis=axis)
    
    # Prepare data
    input_train, output_train, input_test, output_test = prepare_data(input_train, output_train, input_test, output_test)
    
    
    # Fit and predict output(t+t*) from input(t)
    if not runs:
        reservoir_i2o = reservoir_network(I2N, N2N)
        reservoir_i2o.fit(X=input_train, y=output_train)
        output_pred = reservoir_i2o.predict(input_test)
    else:
        output_pred = np.empty(shape=(runs,), dtype=object)
        test_data_copy = np.empty(shape=(runs,), dtype=object)
        for run in range(runs):
            reservoir_i2o = reservoir_network(I2N, N2N)
            reservoir_i2o.fit(X=input_train, y=output_train)
            output_pred[run] = reservoir_i2o.predict(input_test)[0]
            test_data_copy[run] = output_test[0]
        output_test = test_data_copy
    
    # Predictability measured by the correlation between ground-truth and prediction
    correlations = np.zeros((output_test.shape[0],))
    ground_truth, predictions = np.zeros((output_test.shape[0], output_test[0].shape[0]-skip)), np.zeros((output_test.shape[0], output_test[0].shape[0]-skip))
    
    for i, (x, y) in enumerate(zip(output_test, output_pred)):
        correlations[i] = np.corrcoef(x[skip:],y[skip:])[0,1]
        ground_truth[i] = x[skip:]
        predictions[i] = y[skip:]

    return correlations, ground_truth, predictions

def RCC(input, output, lags, I2N, N2N, split=75, skip=20, shuffle=False, axis=0, runs=None):
    """
    TODO: Add description
    """

    # Try different time lags
    results_i2o = pd.DataFrame(columns=["lag","predictability","ground_truth","predictions"])
    results_o2i = pd.DataFrame(columns=["lag","predictability","ground_truth","predictions"])
    for lag in lags:
        # x(t) predicts y(t+t*)
        correlations, ground_truth, predictions = reservoir_input2output(input, output, lag, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis, runs=runs)
        results_i2o.loc[len(results_i2o.index)] = [lag, correlations, ground_truth, predictions]
        
        # y(t) predicts x(t+t*)
        correlations, ground_truth, predictions = reservoir_input2output(output, input, lag, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis, runs=runs)
        results_o2i.loc[len(results_o2i.index)] = [lag, correlations, ground_truth, predictions]

    return results_i2o, results_o2i

def RCC_average(x, y, lags, I2N, N2N, split=75, skip=20, shuffle=False, axis=0, runs=None, average=False):
    """
    TODO: Add description
    """

    # Reservoir Computing Causality - which needs to be tested accross several lags
    results_x2y, results_y2x = RCC(x, y, lags, I2N, N2N, split=split, skip=skip, shuffle=shuffle, axis=axis, runs=runs)

    # We extract the data
    if not runs:
        Nsamples = results_x2y["predictability"][0].shape[0] 
    else:
        Nsamples = runs
    corr_x2y, corr_y2x, sem_x2y, sem_y2x = np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples)), np.zeros((lags.shape[0], Nsamples))
    for i in range(lags.shape[0]):
        corr_x2y[i] = results_x2y["predictability"][i]
        corr_y2x[i] = results_y2x["predictability"][i]
    
    # Stats 
    if average:
        mean_x2y, mean_y2x = np.mean(corr_x2y, axis=1), np.mean(corr_y2x, axis=1)
        sem_x2y, sem_y2x = np.std(corr_x2y, axis=1)/np.sqrt(Nsamples), np.std(corr_y2x, axis=1)/np.sqrt(Nsamples)
        
        return np.expand_dims(mean_x2y, axis=1), np.expand_dims(mean_y2x, axis=1), results_x2y.drop("predictability", axis=1), results_y2x.drop("predictability", axis=1)
    else:
        return corr_x2y, corr_y2x, results_x2y.drop("predictability", axis=1), results_y2x.drop("predictability", axis=1)

def unidirectional_score_ij(p_i2j, p_j2i, p_delta_positive, p_delta_negative, lags):
    """
    TODO: Add description. ONeNote for reference.
    """
    Score_x2y = (lags<0)*(1-p_delta_negative)*(1-p_j2i) + (lags>0)*(1-p_delta_positive)*(1-p_i2j)
    Score_y2x = (lags>0)*(1-p_delta_negative)*(1-p_j2i) + (lags<0)*(1-p_delta_positive)*(1-p_i2j)
    return Score_x2y, Score_y2x
    
def score_ij(p_i2j, p_j2i, p_delta):
    """
    TODO: Add description. ONeNote for reference.
    """
    return (1-p_i2j) * (1-p_j2i) * p_delta

def threshold_unidirectional(p_i2j, p_j2i, p_delta_positive, p_delta_negative, lags, significance=0.05):
    # This score is monotonous with respect to both hypotheses
    # H1: Accept better predictability
    # H2: Accept predictability (considering surrogates)
    # Each direction has its own score
    # Gathering evidence from X -> Y
    evidence_x2y = (lags<0) * (
        # H1 (negative lag)
        ((1-p_delta_negative) > (1-significance)) *
        # H2 (negative lag)
        ((1-p_j2i) > (1-significance))
        ) + (lags>0) * (
        # H1 (positive lag)
        ((1-p_delta_positive) > (1-significance)) * 
        # H2 (positive lag)
        ((1-p_i2j) > (1-significance))
    )
    # Gathering evidence from Y -> X
    evidence_y2x = (lags<0) * (
        # H1 (negative lag)
        ((1-p_delta_positive) > (1-significance)) *
        # H2 (negative lag)
        ((1-p_i2j) > (1-significance))
        ) + (lags>0) * (
        # H1 (positive lag)
        ((1-p_delta_negative) > (1-significance)) * 
        # H2 (positive lag)
        ((1-p_j2i) > (1-significance))
    )        
    return np.where(evidence_x2y==1, 1, np.nan), np.where(evidence_y2x==1, 1, np.nan)

def threshold_bidirectional(p_i2j, p_j2i, p_delta, significance=0.05):
    # This score is monotonous with respect to all three hypotheses
    # We can compute a critical value and anything above is significant
    evidence_xy = (
        # H1: Accept j from i (considering surrogates)
        (1-p_i2j) > (1-significance)
        ) * (
        # H2: Accept i from j (considering surrogates) 
        (1-p_j2i) > (1-significance)
        ) * (
        # H3: Reject delta different from zero
        p_delta > (significance)
    )
    return np.where(evidence_xy==1, 1, np.nan)

def directionality_test_RCC(x2y, y2x, surrogate_x2y, surrogate_y2x, lags, significance=0.05, permutations=False, axis=1, bonferroni=True):
    """
    TODO: Add description. ONeNote for reference.
    """
    
    # Delta: Difference in predictability (McCracken & Weigel Phys. Rev. E. 2014)
    Delta, Delta_surrogate = x2y - y2x, surrogate_x2y - surrogate_y2x
    _, p_delta_positive = ttest_ind(Delta, Delta_surrogate, axis=axis, equal_var=False, permutations=permutations, alternative='greater')
    _, p_delta_negative = ttest_ind(Delta, Delta_surrogate, axis=axis, equal_var=False, permutations=permutations, alternative='less')
    _, p_delta = ttest_ind(Delta, Delta_surrogate, axis=axis, equal_var=False, permutations=permutations, alternative='two-sided')
    
    # Predictabilities are statistically significant
    _, p_x2y = ttest_ind(x2y, surrogate_x2y, axis=axis, equal_var=False, permutations=permutations, alternative='greater')
    _, p_y2x = ttest_ind(y2x, surrogate_y2x, axis=axis, equal_var=False, permutations=permutations, alternative='greater')
    
    # Causality Scores    
    Score_x2y, Score_y2x = unidirectional_score_ij(p_x2y, p_y2x, p_delta_positive, p_delta_negative, lags)
    Score_xy = score_ij(p_x2y, p_y2x, p_delta)
    
    # Statistical evidence: Compute the scores at the critical values (one-sided and two sided respectively)
    # WROOOONG calculations!!!
    """ if bonferroni:
        # Num hypothesis is 3
        threshold_uni, _ = unidirectional_score_ij(significance/2, significance/2, significance/2, significance/2, -1)
        #threshold_bi = score_ij(significance/3, significance/3, significance/(2*3))
        threshold_bi = score_ij(significance/3, significance/3, significance/3)
    else:
        threshold_uni, _ = unidirectional_score_ij(significance, significance, significance, significance, -1)
        #threshold_bi = score_ij(significance, significance, significance/2)
        threshold_bi = score_ij(significance, significance, significance)
    
    evidence_x2y = np.where(Score_x2y>=threshold_uni, 1, np.nan)
    evidence_y2x = np.where(Score_y2x>=threshold_uni, 1, np.nan)
    evidence_xy = np.where(Score_xy>=threshold_bi, 1, np.nan) """
    evidence_x2y, evidence_y2x = threshold_unidirectional(
        p_x2y, p_y2x, p_delta_positive, p_delta_negative, lags, 
        significance=significance/2 if bonferroni else significance
    )
    evidence_xy = threshold_bidirectional(
        p_x2y, p_y2x, p_delta, 
        significance=significance/3 if bonferroni else significance
    )
    return evidence_xy, evidence_x2y, evidence_y2x, Score_xy, Score_x2y, Score_y2x

def directionality_test_GC(data_i2j, data_j2i, lags, significance=0.05, test='F'):
    """
    TODO: Add description. 
    Reference: # https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.grangercausalitytests.html
    """
        
    # To get record results
    R_i2j, R_j2i = np.zeros((len(lags),)), np.zeros((len(lags),))
    evidence_i2j, evidence_j2i = np.zeros((len(lags),)), np.zeros((len(lags),))
    Score_i2j, Score_j2i = np.zeros((len(lags),)), np.zeros((len(lags),))

    # Significance results depend on the test
    assert significance<=1
    significance = 1 - significance
    tests = {'F':'ssr_ftest', 'chi2':'ssr_chi2test', 'lr':'lrtest'}
    if test not in tests.keys():
        raise ValueError("Please provide a valid contrast test: F, chi2, lr")
    else:
        test = tests[test]
    
    # We compute granger tests and extract the significance based 
    for t, lag in enumerate(lags):
        # We only check GC one lag at a time to emulate RCC procedures 
        GC_i2j = grangercausalitytests(data_i2j, maxlag=[lag], verbose=False)[lag]
        GC_j2i = grangercausalitytests(data_j2i, maxlag=[lag], verbose=False)[lag]

        # We extract significance based on pvalues and significance level provided
        R_i2j[t] = np.log1p(GC_i2j[0][test][0])          # Statistic
        R_j2i[t] = np.log1p(GC_j2i[0][test][0])
        Score_i2j[t] = 1 - GC_i2j[0][test][1]  # p-value
        Score_j2i[t] = 1 - GC_j2i[0][test][1] 

    # We binarize the significance
    evidence_i2j = np.where(Score_i2j>=significance, 1, np.nan)
    evidence_j2i = np.where(Score_j2i>=significance, 1, np.nan)

    return R_i2j, R_j2i, evidence_i2j, evidence_j2i, Score_i2j, Score_j2i

def reorder_y2x(data, y, x):
    """
    Re-order the data to test the conditional granger causality from 
        y to x, where x and y are indices of the variables.

    The output data is organized as: data[x], data[y], data[Z], where 
        Z are all the other series.

    Inputs:
        data [T,N]: (np.array) where T is the time samples and N the 
                    number of variables
        y: (int) index of the cause to test
        x: (int) index of the consequence to test
    """
    full = data * 0
    full[:,0] = data[:,x]
    full[:,1] = data[:,y]
    k = 2
    for i in range(data.shape[1]):
        if i!=x and i!=y:
            full[:,k] = data[:,i]
            k += 1
    reduced = np.delete(full, 1, 1)
    return full, reduced

def directionality_test_pwCGC(data, i, j, max_order=5, ic='aic', significance=0.05, test='chi2'):
    """
    # TODO: Add description
    For now using only chi2 test and lag order selection instead of customizeable lag
    """
    # Significance 
    assert significance<=1
    significance = 1 - significance

    # Time samples
    T = data.shape[0]

    ##### i --> j #####
    ts_full, ts_reduced = reorder_y2x(data, i, j)
    # FULL MODEL
    model_full = VAR(ts_full)
    results_full = model_full.fit(maxlags=max_order, ic=ic)
    order_full = results_full.k_ar
    residuals_full = results_full.resid
    res_cov_full = ((residuals_full.T @ residuals_full))/(T-order_full-1)
    # REDUCED MODEL
    model_reduced = VAR(ts_reduced)
    results_reduced = model_reduced.fit(maxlags=max_order, ic=ic)
    order_reduced = results_reduced.k_ar
    residuals_reduced = results_reduced.resid
    res_cov_reduced = ((residuals_reduced.T @ residuals_reduced))/(T-order_reduced-1)
    # STATS
    order = max([order_full, order_reduced])
    F_i2j = np.log(np.abs(res_cov_reduced[0,0])) - np.log(np.abs(res_cov_full[0,0])) # F-value
    Score_i2j = 1 - (1 - chi2.cdf((T-order)*F_i2j, order)) # 1 - P-value
    evidence_i2j = np.where(Score_i2j>=significance, 1, np.nan)


    ##### j --> i #####
    ts_full, ts_reduced = reorder_y2x(data, j, i)
    # FULL MODEL
    model_full = VAR(ts_full)
    results_full = model_full.fit(maxlags=max_order, ic=ic)
    order_full = results_full.k_ar
    residuals_full = results_full.resid
    res_cov_full = ((residuals_full.T @ residuals_full))/(T-order_full-1)
    # REDUCED MODEL
    model_reduced = VAR(ts_reduced)
    results_reduced = model_reduced.fit(maxlags=max_order, ic=ic)
    order_reduced = results_reduced.k_ar
    residuals_reduced = results_reduced.resid
    res_cov_reduced = ((residuals_reduced.T @ residuals_reduced))/(T-order_reduced-1)
    # STATS
    order = max([order_full, order_reduced])
    F_j2i = np.log(np.abs(res_cov_reduced[0,0])) - np.log(np.abs(res_cov_full[0,0])) # F-value
    Score_j2i = 1 - (1 - chi2.cdf((T-order)*F_j2i, order)) # 1 - P-value
    evidence_j2i = np.where(Score_j2i>=significance, 1, np.nan)

    return (
        np.array([F_i2j]), np.array([F_j2i]), 
        np.array([evidence_i2j]), np.array([evidence_j2i]), 
        np.array([Score_i2j]), np.array([Score_j2i])
    )

class Spectral_utils():
    def __init__(self, time_series, max_lags=15, IC='aic', tol=1e-8):
        """
        TODO: Add documentation

        time series is of dimensions T x Regions
        """
        self.time_series = time_series
        self.max_lags = max_lags
        self.ic = IC
        self.tol = tol
        self.var_fit = VAR(time_series).fit(maxlags=max_lags, ic=IC)
        self.A, self. G, self.lags_tol = self.var_to_autocov()

    def var_to_autocov(self):
        """
        Return autocovariance sequence for a VAR model.

        Parameters
        ----------
        VAR : statsmodel object
            Contains all the information of the VAR(p) process fitted with OLS
        tol : float, optional
            Autocovariance decay tolerance. Default is 1e-8.
        Returns
        -------
        A : array_like
            VAR parameters sequence (lag 1, ..., lag p).
        G : array_like
            Autocovariance sequence (lag 1, ..., lag p).
        lags_tol : int
            Actual number of autocovariance lags calculated.

        Notes
        -----
        The function returns the autocovariance sequence `G` for a VAR model with coefficients `A`
        and (positive-definite) residual covariance matrix, by "reverse-solving" the Yule-Walker equations.
        The solution of such equations is given by the statsmodels implementation:
            https://www.statsmodels.org/stable/_modules/statsmodels/tsa/vector_ar/var_model.html#VARProcess.acf
        The residuals are already in the VAR.
        """
        # VAR parameters
        PARAMS = self.var_fit.params[1:] # First index is the mean which should be around zero due to stationarity 
        p, n = PARAMS.shape
        k = p // n
        A = np.reshape(PARAMS, (n,n,k)) # DIMS: n x n x order

        # Number of lags required to achieve the tolerance
        lags_tol = int(
            np.ceil(
                np.log(self.tol) / np.log(
                    np.max(
                        [np.abs(eig(A[..., lag])[0]) for lag in range(k)]
                    )
                )
            )
        )

        # Yule-Walker equations
        G = self.var_fit.acf(nlags=lags_tol)

        return A, G, lags_tol

    def transpose_sequence(self, A):
        p, n, n1 = A.shape
        assert n1 == n, 'Sequence matrix has bad shape'

        A_t = np.zeros((n, n1, p))
        for i in range(p):
            A_t[:,:,i] = A[i,...]

        return A_t
    
    def autocov_to_cpsd(self, G, fres=None): # A9
        """
        Calculate cross-power spectral density from autocovariance sequence.

        Parameters:
        G : ndarray
            Autocovariance sequence.
        fres : int, optional
            Frequency resolution to calculate (default: automatic).

        Returns:
        S : ndarray
            Cross-power spectral density (cpsd) matrix.
        fres : int
            Frequency resolution actually calculated.
        """
        
        n, n1, p = G.shape
        assert n1 == n, 'Autocovariance matrix has bad shape'

        if fres is None:
            fres = G.shape[2]

        h = fres + 1
        G0 = G[:, :, 0]

        # FFT over 2*pi
        freqs = np.fft.fftfreq(n=2*fres)
        S = np.fft.fft(G, n=2*fres, axis=-1)
        
        # over [0, pi]
        freqs = freqs[freqs>=0]
        S = S[:, :, :h]  
        S = S + np.conj(np.transpose(S, (1, 0, 2))) - np.tile(G0[:, :, np.newaxis], (1, 1, h))
        
        return S, freqs
    
    def cpsd_to_autocov(self, S, q=None): # A10
        """
        Calculate autocovariance sequence from cross-power spectral density.

        Parameters
        ----------
        S : array_like
            Cross-power spectral density (cpsd) matrix.
        q : int, optional
            Number of autocovariance lags to calculate. If not supplied, it is set to the frequency resolution of the cpsd.

        Returns
        -------
        G : array_like
            Autocovariance sequence.
        q : int
            Number of autocovariance lags actually calculated.

        Notes
        -----
        Calculates the autocovariance sequence `G` to `q` lags from the cross-power spectral density (cpsd) `S`. 
        This is essentially an inverse Fourier transform implemented as an (discrete) inverse fast Fourier transform. 
        If a number of lags `q` is not supplied, then the default is to set it to the frequency resolution of the cpsd.
        The actual number of lags calculated is returned in `q`.
        """

        n, _, h = S.shape
        fres = h - 1
        if q is None:
            q = fres - 1
        assert q < 2 * fres, 'Too many lags'
        
        if q%2 != 0:
            q1 = q + 1
        else:
            q1 = q

        # Inverse transform of "circular shifted" spectral density (Eq. A15 Barnett & Seth, 2015)
        S_conj_flip = np.flip(np.conj(S[:, :, 1:fres+1]), axis=2)
        S_concat = np.concatenate((S_conj_flip, S[:, :, :fres]), axis=2)
        G = np.fft.ifft(S_concat, n=2*fres, axis=2)
        
        """ r = np.ones((1, int(np.ceil(q1 / 2))))
        sgn = np.vstack((r, -r)).flatten()[:q1]
        sgn = np.tile(sgn, (n * n, 1)) """

        # Create an array of ones with the size of half of q1, rounded up
        r = np.ones(int(np.ceil(q1 / 2)))

        # Create the alternating sign sequence
        sgn = np.empty(q1)
        sgn[0::2] = r  # Fill even indices with 1
        sgn[1::2] = -r  # Fill odd indices with -1

        G = np.real(np.reshape(sgn * np.reshape(G[:, :, :q1], (n * n, q1)), (n, n, q1)))

        return G, q

    def var_to_transfer(self, A, fres=None):
        """
        Calculate VAR transfer function from VAR coefficients

        Syntax
        ------
            H = var2trfun(A, fres)

        Arguments
        ---------
        See also Common variable names and data structures.

        Input
        -----
        A : ndarray
            VAR coefficients matrix
        fres : int
            Frequency resolution

        Output
        ------
        H : ndarray
            VAR transfer function matrix

        Description
        -----------
        Return transfer function `H` for VAR with coefficients `A`. `fres` specifies the frequency resolution. Call 
        `freqs = sfreqs(fres, fs)`, where `fs` is the sampling rate, to get a corresponding vector `freqs` of frequencies on `[0, fs/2]`.
        """

        n, n1, p = A.shape
        assert n1 == n, 'VAR matrix has bad shape'

        if fres is None:
            fres = A.shape[2]

        # Fourier transform of the VAR
        #I = np.eye(n)
        #AF = I[..., np.newaxis] - A 
        #AF = np.fft.fft(AF, 2*fres)
        AF = np.fft.fft(np.concatenate((np.eye(n)[..., np.newaxis], -A), axis=2), 2*fres)
        
        # Transfer function: Inverse matrix of the Fourier transform (Eq. 10; Barnett & Seth, 2015)
        h = fres + 1
        H = np.zeros((n, n, h), dtype=complex)
        for k in range(h):  # over [0, pi] only
            H[:, :, k] = np.linalg.inv(AF[:, :, k])
        
        return H

    def autocov_to_var(self, G):
        """
        Calculate VAR parameters from autocovariance sequence.

        Syntax
        ------
            A, SIG = autocov_to_var(G)

        Arguments
        ---------
        See also Common variable names and data structures.

        Input
        -----
        G : ndarray
            Autocovariance sequence

        Output
        ------
        A : ndarray
            VAR coefficients matrix
        SIG : ndarray
            Residuals covariance matrix

        Description
        -----------
        Calculates regression coefficients `A` and residuals covariance matrix `SIG` from the autocovariance sequence `G`
        by solving the Yule-Walker equations. For a `q`-lag autocovariance sequence, this routine corresponds to an 
        autoregression of `q` lags. It also effects an efficient spectral factorisation if called with the 
        autocovariance sequence derived from the cross-power spectral density.

        This routine implements Whittle's recursive LWR algorithm which, for `n` variables, performs `2q` separate 
        `n x n` matrix inversions as compared with a single `nq x nq` matrix inversion for the conventional "OLS" 
        solution of the Yule-Walker equations. The LWR algorithm also (unlike OLS) guarantees that if the "true" 
        regression model is stable, then the estimated model is also stable, EVEN IF NOT OF THE CORRECT ORDER!  

        Note: If the regressions are rank-deficient or ill-conditioned then A may be "bad" (i.e. will contain a `NaN` 
        or `Inf`) and/or warnings may be issued. The caller should test for both these possibilities.
        """

        n, n1, q1 = G.shape
        assert n1 == n, 'Autocovariance matrix has bad shape'

        q = q1 - 1
        qn = q * n
        G0 = G[:, :, 0]  # covariance
        GF = np.reshape(G[:, :, 1:], (n, qn), order='F').T  # forward autocov sequence
        GB = np.reshape(np.transpose(np.flip(G[:, :, 1:], axis=2), (2, 0, 1)), (qn, n))

        AF = np.zeros((n, qn))  # forward coefficients
        AB = np.zeros((n, qn))  # backward coefficients (reversed compared with Whittle's treatment)

        # Initialize recursion
        k = 1  # model order
        r = q - k
        kf = slice(0, k * n)  # forward indices
        kb = slice(r * n, qn)  # backward indices
        AF[:, kf] = np.linalg.solve(G0, GB[kb, :].T).T
        AB[:, kb] = np.linalg.solve(G0, GF[kf, :].T).T

        # Loop
        for k in range(2, q + 1):   
            AAF = np.linalg.solve(
                G0 - AB[:, kb] @ GB[kb, :], 
                (GB[(r - 1) * n:r * n, :] - AF[:, kf] @ GB[kb, :]).T
            ).T
            
            AAB = np.linalg.solve(
                G0 - AF[:, kf] @ GF[kf, :], 
                (GF[(k - 1) * n:k * n, :] - AB[:, kb] @ GF[kf, :]).T
            ).T

            AFPREV = AF[:, kf]
            ABPREV = AB[:, kb]

            r = q - k
            kf = slice(0, k * n)
            kb = slice(r * n, qn)        
            
            AF[:, kf] = np.hstack((AFPREV - AAF @ ABPREV, AAF))
            AB[:, kb] = np.hstack((AAB, ABPREV - AAB @ AFPREV))

        SIG = G0 - AF @ GF
        AF = AF.reshape(n, n, q, order='F')
        """ AF_transpose = AF.copy()
        for k in range(AF.shape[-1]):
            AF_transpose[...,k] = AF[...,k].T """
        
        return AF, SIG
    
    def autocov_xform(self, G, AR, SIGR):
        """
        Transform autocovariance sequence for reduced regression

        Syntax
        ------
            G = autocov_xform(G, AR, SIGR, useFFT=True)

        Arguments
        ---------
        See also Common variable names and data structures.

        Input
        -----
        G : ndarray
            Autocovariance sequence
        AR : ndarray
            VAR coefficients matrix for reduced regression
        SIGR : ndarray
            Residuals covariance matrix for reduced regression

        Output
        ------
        G : ndarray
            Transformed autocovariance sequence

        Description
        -----------
        Returns the autocovariance sequence `G` for a new variable defined as the residuals of a reduced regression, 
        for a VAR with autocovariance sequence `G`. `AR` and `SIGR` are the coefficients matrices and residuals 
        covariance matrix respectively of the reduced regression, which is assumed to correspond to the first 
        `size(AR, 1)` indices of `G`.

        If the `useFFT` flag is set (default), then the autocovariance sequence is converted to a cpsd via FFT, 
        the transformation effected on the cpsd, and the result converted back to an autocovariance sequence via IFFT.
        Otherwise, the autocovariance sequence is transformed by explicit convolution. The FFT method is generally 
        more efficient than the convolution method, particularly if the number of autocovariance lags is large.

        This function is crucial to the calculation of spectral causality in the conditional case. In theory, if the 
        original autocovariance sequence is calculated to `q` lags, then the transformed autocovariance sequence should 
        be calculated to `2q` lags. In practice, calculating to `q` lags is generally sufficient for good accuracy. 
        To calculate `G` to higher lags, the simplest option is to reduce the `acdectol` parameter.
        """

        n, _, q1 = G.shape
        #q = q1 - 1
        nx, nx1, _ = AR.shape

        assert nx1 == nx, 'Reduced VAR coefficients matrix has bad shape'
        assert nx <= n, 'Reduced VAR coefficients need to be smaller'
        
        n1, n2 = SIGR.shape
        assert n1 == n2, 'Reduced VAR residuals covariance matrix not square'
        assert n1 == nx, 'Reduced VAR residuals covariance matrix doesn\'t match reduced VAR coefficients matrix'
        
        ny = n - nx
        """ x = slice(nx)
        y = slice(nx, n) """
        x = np.arange(nx)
        y = np.arange(nx, n)
        
        S, _ = self.autocov_to_cpsd(G)
        
        AF = np.fft.fft(np.concatenate((np.eye(nx)[..., np.newaxis], -AR), axis=2), 2*q1)
        #AF = bifft(np.concatenate((np.eye(nx)[..., np.newaxis], -AR[..., np.newaxis]), axis=2), 2 * q1)
        
        for k in range(q1 + 1): # Eq. 21 Barnett & Seth, 2015
            S[x[:, np.newaxis], x, k] = SIGR
            S[x[:, np.newaxis], y, k] = np.dot(AF[:, :, k], S[x[:, np.newaxis], y, k])
            S[y[:, np.newaxis], x, k] = np.dot(S[y[:, np.newaxis], x, k], np.conj(AF[:, :, k]).T)

        G, _ = self.cpsd_to_autocov(S)
        
        return G
    
    def autocov_to_spwcgc(self, fres=None):
        """
        Calculate pairwise-conditional frequency-domain MVGCs (spectral multivariate Granger causalities)

        This function computes the matrix of pairwise-conditional frequency-domain (spectral) multivariate Granger causalities (MVGCs).

        Functions
        ---------
        autocov_to_spwcgc(G, fres=None, useFFT=None)
            Calculate the spectral Granger causality matrix.

        Arguments
        ---------
        G : array-like
            Autocovariance sequence.
        fres : int, optional
            Frequency resolution. If not supplied, it is calculated optimally as the number of autocovariance lags.

        Returns
        -------
        f : array-like
            Spectral Granger causality matrix.

        Description
        -----------
        This function returns the matrix `f` of pairwise-conditional frequency-domain (spectral) MVGCs between all pairs of variables represented in `G`, for a stationary VAR process with autocovariance sequence `G`. The first index `i` of `f` is the target (causee) variable, the second `j` the source (causal) variable, and the third indexes the frequency. See Barnett and Seth [1] for details.

        Spectral causality is calculated up to the Nyquist frequency at a resolution `fres`. If `fres` is not supplied, it is calculated optimally as the number of autocovariance lags. Call `freqs = sfreqs(fres, fs)` to get a corresponding vector `freqs` of frequencies on `[0, fs/2]`.

        The `useFFT` flag specifies the algorithm used to transform the autocovariance sequence. See `autocov_xform` for details.

        The caller should take note of any warnings issued by this function and test results with a call to `isbad(f, False)`.

        For details of the algorithm, see `autocov_to_smvgc` and [1].

        References
        ----------
        [1] L. Barnett and A. K. Seth,
            "The MVGC Multivariate Granger Causality Toolbox: A New Approach to Granger-causal Inference", J. Neurosci. Methods 223, 2014
            [Preprint](mvgc_preprint.pdf)

        See Also
        --------
        autocov_to_smvgc
        autocov_to_pwcgc
        autocov_to_var
        var2trfun
        autocov_xform
        sfreqs
        isbad

        (C) Lionel Barnett and Anil K. Seth, 2012. See file license.txt in installation directory for licensing terms.
        """
        
        try:
            n, n1, q1 = self.G.shape
            assert n==n1, 'WARNING: The dimensions of the autocovariance sequence do not match the expected, attempting to transpose it'
        except:
            self.G = self.transpose_sequence(self.G)
            n, n1, q1 = self.G.shape
            assert n==n1, 'The dimensions of the autocovariance sequence are wrongly ordered'
        
        if fres is None:
            fres = q1
        
        h = fres + 1
        f = np.full((n, n, h), np.nan)
        
        for j in range(n):
            # Reduced
            jo = [i for i in range(n) if i != j] 
            G_jo = self.G[np.ix_(jo, jo, np.arange(self.G.shape[2]))]

            # Full rearranged
            joj = jo + [j] 
            G_joj = self.G[np.ix_(joj, joj, np.arange(self.G.shape[2]))]

            Aj, SIGj = self.autocov_to_var(G_jo)  # reduced regression
            Gj = self.autocov_xform(G_joj, Aj, SIGj)  # transform autocov
            Ajj, SIGjj = self.autocov_to_var(Gj) # reduced VAR
            Hjj = self.var_to_transfer(Ajj, fres) # Transfer function of the reduced var

            for ii in range(n - 1):
                i = jo[ii]  # i index in omitted j indices
                io = [ss for ss in range(n) if ss != ii] # omit i
                
                SIGji = SIGjj[np.ix_(io, io)] - (np.outer(SIGjj[io, ii], SIGjj[ii, io]) / SIGjj[ii, ii])  # partial covariance
                Hji = Hjj[ii, io, :]  # transfer function
                Sji = SIGj[ii, ii]  # i part of spectrum is flat!
                LSji = np.log(Sji)
                
                for k in range(h):
                    f[j, i, k] = LSji - np.log(np.abs(Sji - Hji[:, k] @ SIGji @ np.conj(Hji[:, k]).T))
        
        return f, fres
    
    def spwcgcm_to_cgm(self, f, fres=None, max_freq=None):
        """
        Average (integrate) frequency-domain causality over specified frequency range.

        Syntax
        ------
            F = scgcm_to_cgm(f, fres=None, max_freq=np.pi())

        Arguments
        ---------
        f : array-like
            Spectral (frequency-domain) Granger causality.
        fres : Points to integrate along. X axis resolution.
        max_freq: Frequency maximum range specified by pairs of points in the range [0, 2*pi]. 
                  If unspecified (default), the entire frequency range is used.

        Returns
        -------
        F : array-like
            Granger causality (time domain).
        xx_[...]: array-like
                  points along which the integration has been carried out

        Description
        -----------
        Calculates (conditional or unconditional) time-domain causality `F` from
        spectral causality `f` by integration (numerical quadrature) over the frequency 
        range `B`. If a frequency band `B` is not supplied (default), the spectral 
        causality is averaged from zero to the Nyquist frequency. In that case, the 
        formula should hold (at least approximately, numerically).

        A frequency band `B` is specified by a list comprising pairs of points in 
        ascending order in the range [0, 1] - corresponding to zero up to the Nyquist 
        frequency. In this case, band-limited time-domain causality is calculated.

        References
        ----------
        [1] L. Barnett and A. K. Seth,
            "The MVGC Multivariate Granger Causality Toolbox: A New Approach to 
            Granger-causal Inference", J. Neurosci. Methods 223, 2014.
            (http://www.sciencedirect.com/science/article/pii/S0165027013003701)

        [2] L. Barnett and A. K. Seth, 
            "Behaviour of Granger causality under filtering: Theoretical invariance 
            and practical application", J. Neurosci. Methods 201(2), 2011.

        See Also
        --------
        quads : Numerical integration using quadrature.
        quadsr : Numerical integration over sub-ranges.
        sfreqs : Calculate the corresponding vector of frequencies.
        mvgc_demo : Demonstration of the MVGC Toolbox.

        (C) Lionel Barnett and Anil K. Seth, 2012. See file license.txt in
        installation directory for licensing terms.
        """
        n, n1, q1 = f.shape
        assert n==n1, 'WARNING: The dimensions of the granger causality measure sequence do not match the expected, attempting to transpose it'
        
        # Default settings
        if (fres is None) and (max_freq is None):
            fres = f.shape[-1]
            xx = np.linspace(0, 2*np.pi, fres)
            
            F = cumulative_simpson(f, x=xx, axis=-1)[...,-1]
            return F, xx

        # User defined resolution until 2*pi
        if (fres is not None) and (max_freq is None):
            xx = np.linspace(0, 2*np.pi, f.shape[-1])
            xx_interp = np.linspace(0, 2*np.pi, fres)

            f_interp = np.zeros((n, n1,fres))
            for i, j in np.ndindex(f[...,0].shape):
                f_interp[i,j] = np.interp(xx_interp, xx, f[i,j,:], left=f[i,j,0], right=f[i,j,-1])

            F = cumulative_simpson(f_interp, x=xx_interp, axis=-1)[...,-1]

            return F, xx_interp

        # User-defined resolution and upper limit
        if (fres is not None) and (max_freq is not None):
            xx = np.linspace(0, 2*np.pi, f.shape[-1])
            xx_interp = np.linspace(0, 2*np.pi, fres)

            f_interp = np.zeros((n, n1, fres))
            for i, j in np.ndindex(f[...,0].shape):
                f_interp[i,j] = np.interp(xx_interp, xx, f[i,j,:], left=f[i,j,0], right=f[i,j,-1])

            xx_interp_lim = xx_interp[xx_interp<=max_freq+.1]
            f_interp_lim = f_interp[...,xx_interp<=max_freq+.1]

            F = cumulative_simpson(f_interp_lim, x=xx_interp_lim, axis=-1)[...,-1]

            return F, xx_interp_lim
        
        # Default settings
        if (fres is None) and (max_freq is not None):
            fres = f.shape[-1]
            xx = np.linspace(0, 2*np.pi, fres)

            xx_lim = xx[xx<=max_freq+.1]
            f_lim = f[...,xx<=max_freq+.1]
            
            F = cumulative_simpson(f_lim, x=xx_lim, axis=-1)[...,-1]
            return F, xx_lim
        
def directionality_test_spwcgc(data, t_surrogates, n_ROIS, max_order=15, ic='aic', significance=0.05, tol=1e-8):
    """
    TODO: Add documentation
    """

    # Real data
    spwcgc = Spectral_utils(data, max_lags=max_order, IC=ic, tol=tol)
    _, G, _ = spwcgc.var_to_autocov()
    G = spwcgc.transpose_sequence(G)
    S, _ = spwcgc.autocov_to_cpsd(G)
    f, _ = spwcgc.autocov_to_spwcgc()
    F, x_axis = spwcgc.spwcgcm_to_cgm(f)

    # All surrogates
    N_surrogates = t_surrogates.shape[1]
    F_surrs = np.zeros((N_surrogates,n_ROIS,n_ROIS))
    f_surrs = np.empty(shape=(N_surrogates,), dtype=object)
    x_surrs = np.empty(shape=(N_surrogates,), dtype=object)
    for s in range(N_surrogates):
        spcgc_surr = Spectral_utils(t_surrogates[:,s,:].T, max_lags=max_order, IC=ic, tol=tol)
        fs, _ = spcgc_surr.autocov_to_spwcgc()
        Fs, xs = spcgc_surr.spwcgcm_to_cgm(fs)

        F_surrs[s] = Fs
        f_surrs[s] = fs
        x_surrs[s] = xs

    # Non-parametric test
    p_val_F = (np.abs(F_surrs) >= np.abs(F)).mean(axis=0)  + np.eye(n_ROIS) # not including the diagonal (i.e., p(1,1)=1)
    evidence_F = np.where(p_val_F<=significance, 1, 0)

    return ((F, f, x_axis), (F_surrs, f_surrs, x_surrs), (p_val_F, evidence_F))

if __name__ == '__main__':
    pass