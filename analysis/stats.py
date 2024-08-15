import numpy as np
from sklearn.utils import resample
from sklearn.metrics import roc_curve, roc_auc_score, det_curve
import scipy.stats
import matplotlib.pylab as plt

# Relative imports
from analysis.utils import RCC_Scores

class Bootstrapping():
    def __init__(self, ground_truth, random_state=None) -> None:
        """
        Docs
        """
        self.ground_truth = ground_truth
        self.random_state = random_state

    def bootstrap_sample(self, sample, metric, n_bootstrap=1000):
        # Bootstrap samples
        metric_boot = np.zeros((n_bootstrap,))
        i=0
        while i<n_bootstrap:
            pred_boot, gt_boot = resample(
                sample, self.ground_truth, replace=True,
                n_samples=len(sample), random_state=self.random_state
            )
            if len(np.unique(gt_boot))==2:
                # At least one sample of each class is needed
                metric_boot[i] = metric(gt_boot, pred_boot)#.ravel()
                i += 1
            else:
                continue
        
        # Sort metric to obtain the lower and upper thresholds
        return np.sort(metric_boot)
    
    def bootstrap_dataset(self, samples, metric, n_bootstrap=1000):
        metric_boot = np.zeros((samples.shape[0], n_bootstrap))            
        for i in range(samples.shape[0]):
            metric_boot[i,:] = self.bootstrap_sample(
                samples[i], metric, n_bootstrap=n_bootstrap
            )
        return metric_boot
    
    def compute_stats(self, object, results_boot, level=95):
        n_bootstrap = results_boot.shape[-1]
        object.all = results_boot
        # Central measures
        object.mean_boot = results_boot.mean(axis=-1)
        object.median_boot = np.median(results_boot, axis=-1)
        # Dispersion measures
        object.std = results_boot.std(axis=-1)
        object.sem =  object.std / np.sqrt(n_bootstrap)
        
        # Significance level
        alpha = (100 - level)/200
        if alpha > 1 or alpha < 0:
            raise ValueError("Provide a Confidence Level between 0 and 100")
        
        # Non-parametric bounds
        ID_lower = int(alpha * n_bootstrap)
        ID_upper = int((1-alpha) * n_bootstrap)

        # Parametric bounds
        z_stat = scipy.stats.norm.ppf(1-alpha) if n_bootstrap>=30 else scipy.stats.t.ppf(1-alpha, df=n_bootstrap-1)

        # Confidence Interval
        object.CI = RCC_Scores()
        object.CI.parametric, object.CI.nonparametric = RCC_Scores(), RCC_Scores()
        object.CI.parametric.std = (object.mean_boot - z_stat * object.std, object.mean_boot + z_stat * object.std)
        object.CI.parametric.sem = (object.mean_boot - z_stat * object.sem, object.mean_boot + z_stat * object.sem)
        object.CI.nonparametric = (results_boot.T[ID_lower], results_boot.T[ID_upper])

        # TODO: Tolerance Intervals
        return object

class DET_utils():
    # Adapted from roc_utils: https://github.com/hirsch-lab/roc-utils/tree/main

    def __init__(self, predictions, ground_truths) -> None:
        
        if list(np.unique(predictions)) == [0,1]:
            raise ValueError("Please provide a weighted network, not binary")
        
        self.predictions = predictions
        self.ground_truths = ground_truths

        # Compute DET curve for real data
        self.fpr_real, self.fnr_real, self.thresholds_real = det_curve(self.ground_truths, self.predictions)

        # TODO: Think whether we can and want to bootstrap this curve

    def plot(self, sample_auc, color="black", save: str=None, dpi: int=None):
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        
        ax.plot(scipy.stats.norm.ppf(self.fpr_real), scipy.stats.norm.ppf(self.fnr_real), c=color, linewidth=2, label=f"AUC={round(sample_auc,3)}")
        ax.plot(scipy.stats.norm.ppf(self.fpr_real), -scipy.stats.norm.ppf(self.fpr_real), c="black", linewidth=0.5, linestyle='--', label="Chance levels")
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False), ax.spines["right"].set_visible(False)
        ax.set_xlabel("False Positive Rate", fontsize=15)
        ax.set_ylabel("False Negative Rate", fontsize=15)
        
        ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
        tick_locations = scipy.stats.norm.ppf(ticks)
        tick_labels = [
            "{:.0%}".format(s) if (100 * s).is_integer() else "{:.1%}".format(s)
            for s in ticks
        ]
        ax.set_xticks(tick_locations)
        ax.set_yticks(tick_locations)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

        if save is not None:
            format = save.split(".")[-1]
            if dpi is not None:
                plt.savefig(save, dpi=dpi, format=format)
            else:
                plt.savefig(save, format=format) 
        else:
            plt.show()
        return fig

class ROC_utils():
    # Adapted from roc_utils: https://github.com/hirsch-lab/roc-utils/tree/main

    def __init__(self, predictions, ground_truths, drop_intermediate=False) -> None:
        
        if list(np.unique(predictions)) == [0,1]:
            raise ValueError("Please provide a weighted network, not binary")  
                                         
        self.predictions = predictions
        self.ground_truths = ground_truths
        self.drop_intermediate = drop_intermediate

        # Compute ROC curve for real data
        self.fpr_real, self.tpr_real, self.thresholds_real = roc_curve(self.ground_truths, self.predictions, drop_intermediate=self.drop_intermediate)

    def bootstrap_roc(self, n_bootstrap=1000, CI_level=95, random_state=None, resolution=0.1):
        """
        Documentation
        """

        # Relevant quantities
        fpr_interp = np.linspace(0, 1, int(1/resolution))
        tpr_interp = np.zeros((n_bootstrap, fpr_interp.shape[0]))
        # TODO: Add other metrics -- if applicable
        auc_boot = np.zeros((n_bootstrap,))

        # Significance level
        alpha = (100 - CI_level)/200
        
        # Non-parametric bounds
        ID_lower = int(alpha * n_bootstrap)
        ID_upper = int((1-alpha) * n_bootstrap)

        # Parametric bounds
        z_stat = scipy.stats.norm.ppf(1-alpha) if n_bootstrap>=30 else scipy.stats.t.ppf(1-alpha, df=n_bootstrap-1)

        # Bootstrapping
        i = 0
        while i<n_bootstrap:
            pred_boot, gt_boot = resample(
                self.predictions, self.ground_truths, replace=True, 
                n_samples=len(self.predictions), random_state=random_state
            )
            if len(np.unique(gt_boot))==2:
                # At least one sample of each class is needed
                fpr_boot, tpr_boot, _ = roc_curve(gt_boot, pred_boot, drop_intermediate=self.drop_intermediate)
                tpr_interp[i] = np.interp(fpr_interp, fpr_boot, tpr_boot)
                tpr_interp[i,0] = 0
                auc_boot[i] = roc_auc_score(gt_boot, pred_boot)
                i += 1
            else:
                continue

        # Sort based AUCs to obtain the lower and upper thresholds
        sorted_aucs = np.copy(auc_boot)
        sorted_aucs.sort()
        x = auc_boot.argsort()
        tpr_interp_sorted_auc = np.zeros((n_bootstrap, fpr_interp.shape[0]))
        for i,j in enumerate(x):
            tpr_interp_sorted_auc[i,:] = tpr_interp[j,:]

        # Sorting based on true positive rates --> this ensures that curves don't cross
        #   but it requires a re-ordering that creates curves that were not directly bootstrapped
        tpr_interp_sorted_tpr = np.sort(tpr_interp, axis=0)

        # Statistics from the bootstrapped distribution
        ROC_BOOTSTRAPPED = RCC_Scores()
        ROC_BOOTSTRAPPED.fpr = fpr_interp
        ROC_BOOTSTRAPPED.tpr = tpr_interp

        # Central measures
        ROC_BOOTSTRAPPED.mean_tpr = tpr_interp.mean(axis=0)
        ROC_BOOTSTRAPPED.median_tpr = np.median(tpr_interp, axis=0)
        # Dispersion measures
        ROC_BOOTSTRAPPED.std = tpr_interp.std(axis=0)
        ROC_BOOTSTRAPPED.sem =  ROC_BOOTSTRAPPED.std / np.sqrt(n_bootstrap)
        # Confidence Interval
        ROC_BOOTSTRAPPED.CI = RCC_Scores()
        ROC_BOOTSTRAPPED.CI.parametric, ROC_BOOTSTRAPPED.CI.nonparametric = RCC_Scores(), RCC_Scores()
        ROC_BOOTSTRAPPED.CI.parametric.true_value = (ROC_BOOTSTRAPPED.mean_tpr - z_stat * ROC_BOOTSTRAPPED.std, ROC_BOOTSTRAPPED.mean_tpr + z_stat * ROC_BOOTSTRAPPED.std)
        ROC_BOOTSTRAPPED.CI.parametric.mean_bootstrap = (ROC_BOOTSTRAPPED.mean_tpr - z_stat * ROC_BOOTSTRAPPED.sem, ROC_BOOTSTRAPPED.mean_tpr + z_stat * ROC_BOOTSTRAPPED.sem)
        ROC_BOOTSTRAPPED.CI.nonparametric.sorted_auc = (tpr_interp_sorted_auc[ID_lower,:], tpr_interp_sorted_auc[ID_upper,:])
        ROC_BOOTSTRAPPED.CI.nonparametric.sorted_tprates = (tpr_interp_sorted_tpr[ID_lower,:], tpr_interp_sorted_tpr[ID_upper,:])
        # Tolerance Intervals
        # TODO
        # https://github.com/jg-854/tolerance_intervals/blob/master/tolerances.py
        # https://math.stackexchange.com/questions/3724039/tolerance-limit-interval-in-python
        self.ROC_BOOTSTRAPPED = ROC_BOOTSTRAPPED
        return ROC_BOOTSTRAPPED

    def plot(self, sample_auc, color="black", save: str=None, dpi: int=None, figure=None):
        if figure is None:
            fig, ax = plt.subplots(1,1,figsize=(6,5))
        else:
            fig, ax = figure   
        if self.ROC_BOOTSTRAPPED is None:
            raise ValueError("Please run bootstrap before plotting")
        
        ax.plot(self.fpr_real, self.tpr_real, c=color, linewidth=2, label=f"AUC={round(sample_auc,3)}")
        ax.plot(self.fpr_real, self.fpr_real, c="black", linewidth=0.5, linestyle='--', label="Chance levels")
        ax.fill_between(self.ROC_BOOTSTRAPPED.fpr, *self.ROC_BOOTSTRAPPED.CI.nonparametric.sorted_tprates, color=color, alpha=0.1, edgecolor=None)
        for i in range(100):
            ax.plot(self.ROC_BOOTSTRAPPED.fpr, self.ROC_BOOTSTRAPPED.tpr[i], alpha=0.05, c="gray") 
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False), ax.spines["right"].set_visible(False)
        ax.set_xlabel("False Positive Rate", fontsize=15)
        ax.set_ylabel("True Positive Rate", fontsize=15)
        ax.set_xlim([0,1.01])
        ax.set_ylim([0,1.01])
        ax.set_xticks([0,0.25,0.5,0.75,1])
        ax.set_yticks([0,0.25,0.5,0.75,1])

        if save is not None:
            format = save.split(".")[-1]
            if dpi is not None:
                plt.savefig(save, dpi=dpi, format=format)
            else:
                plt.savefig(save, format=format) 
        else:
            plt.show()
        return fig, ax

class DeLong_Test():    
    # Adopted from https://github.com/yandexdataschool/roc_comparison 
    # Original ref: https://ieeexplore.ieee.org/document/6851192

    def __init__(self, ground_truths) -> None:
        """
        Computes the DeLong p-value and/or variance of a pair or predictions
            that are related to the same structure (or ground truth)
        Args:
        ground_truths: A flat array containing the positive and negative samples.
        """
        self.ground_truth = ground_truths

    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    def __compute_midrank(self, x):
        """Computes midranks.
        Args:
        x - a 1D numpy array
        Returns:
        array of midranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N, dtype=np.float64)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5*(i + j - 1)
            i = j
        T2 = np.empty(N, dtype=np.float64)
        # Note(kazeevn) +1 is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    def __compute_ground_truth_statistics(self):
        assert np.array_equal(np.unique(self.ground_truth), [0, 1])
        order = (-self.ground_truth).argsort()
        label_1_count = int(self.ground_truth.sum())
        return order, label_1_count

    def fastDeLong(self, predictions_sorted_transposed, label_1_count):
        """
        The fast version of DeLong's method for computing the covariance of
        unadjusted AUC.
        Args:
        predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
            sorted such as the examples with label "1" are first
        Returns:
        (AUC value, DeLong covariance)
        Reference:
        @article{sun2014fast,
        title={Fast Implementation of DeLong's Algorithm for
                Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
        author={Xu Sun and Weichao Xu},
        journal={IEEE Signal Processing Letters},
        volume={21},
        number={11},
        pages={1389--1393},
        year={2014},
        publisher={IEEE}
        }
        """
        # Short variables are named as they are in the paper
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=np.float64)
        ty = np.empty([k, n], dtype=np.float64)
        tz = np.empty([k, m + n], dtype=np.float64)
        for r in range(k):
            tx[r, :] = self.__compute_midrank(positive_examples[r, :])
            ty[r, :] = self.__compute_midrank(negative_examples[r, :])
            tz[r, :] = self.__compute_midrank(predictions_sorted_transposed[r, :])
        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
        sx = np.cov(v01)
        sy = np.cov(v10)
        delongcov = sx / m + sy / n
        return aucs, delongcov
    
    def calc_pvalue(self, aucs, sigma, alternative="two-sided"):
        """Computes the p-value.
        Args:
        aucs: 1D array of AUCs
        sigma: AUC DeLong covariances
        Returns:
        z_score, p_value
        """
        """ l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
        print(z)
        return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10) """     
        if alternative not in ["two-sided", "greater", "lower"]:
            raise ValueError("Provide a valid hypothesis from two-sided, greater or lower")
        l = np.array([1, -1])
        z = (aucs[0]-aucs[1]) / np.sqrt(np.dot(np.dot(l, sigma), l.T))   
        if alternative=="two-sided":
            return z, scipy.stats.norm.sf(abs(z))*2
        elif alternative=="greater":
            return z, scipy.stats.norm.sf(z)
        else:
            return z, scipy.stats.norm.cdf(z)

    def delong_roc_variance(self, predictions):
        """
        Computes ROC AUC variance for a single set of predictions
        Args:
        ground_truth: np.array of 0 and 1
        predictions: np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.__compute_ground_truth_statistics()
        predictions_sorted_transposed = predictions[np.newaxis, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
        return aucs[0], delongcov

    def delong_roc_test(self, predictions_one, predictions_two, alternative="two-sided"):
        """
        Computes log(p-value) for hypothesis that two ROC AUCs are different
        Args:
        ground_truth: np.array of 0 and 1
        predictions_one: predictions of the first model,
            np.array of floats of the probability of being class 1
        predictions_two: predictions of the second model,
            np.array of floats of the probability of being class 1
        """
        order, label_1_count = self.__compute_ground_truth_statistics()
        predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
        aucs, delongcov = self.fastDeLong(predictions_sorted_transposed, label_1_count)
        return self.calc_pvalue(aucs, delongcov, alternative=alternative)