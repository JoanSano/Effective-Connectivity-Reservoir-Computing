import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings

# Relative imports
from analysis.utils import RCC_Scores
from analysis.stats import Bootstrapping

def create_logistic_networks(lags: list, x2y: dict, y2x: dict) -> dict:
    networks_weighted, networks_binary = dict(), dict()
    for lag in lags:
        net = np.zeros((2,2))
        # x --> y [tau]
        if lag in x2y.keys():
            net[0,1] = x2y[lag]
        # y --> x [tau]
        if lag in y2x.keys():
            net[1,0] = y2x[lag]
        # Network [tau]
        networks_weighted[lag] = net
        networks_binary[lag] = np.where(net>0, 1, 0)

    return networks_weighted, networks_binary

def load_netsim_networks(netsim_dir, sim, subject_list):
    networks_weighted, networks_binary = dict(), []
    for i, s in enumerate(subject_list):
        p = os.path.join(netsim_dir, "Networks/") + s + "_" + sim.lower() + "_Net.txt"
        net = np.genfromtxt(p)
        networks_weighted[s] = net + np.eye(net.shape[0])
        networks_binary.append(np.where(net>0,1,0))
    # All binary networks are identical
    networks_binary = np.array(networks_binary).mean(axis=0) 
    return networks_weighted, networks_binary

class Metrics(Bootstrapping):
    def __init__(self, ground_truth, weighted_predictions=None, binary_predictions=None, random_state=None) -> None:
        """
        The predictions and ground truth have to be flattened and constraint (if applicable) by the Constraints class 
            in the analysis.utils module. Alternatively, the wrappers classes at analysis.utils.ground_truths can 
            also be employed.

        The predictions and ground truth correspond to the 1D array of possible connections for a given structure or
            set of allowed connections. Alternatively, they can be an array of samples with multiple instances of a 
            classification and/or classifier. Importantly, they all need to be in correspondance with the same
            ground truth.

        Args:
        weighted_predictions : array_like {np.array(n_samples, n_connections), np.array(n_connections,)}
                            Contains the scores - bounded between 0 and 1 - for each connection
        binary_predictions : array_like  {np.array(n_samples, n_connections), np.array(n_connections,)}
                            Contains the true binary scores (0 or 1) for each connection
        ground_truth : array_like  {np.array(n_connections,)}
                        Contains the true connections (0 or 1) for each connection
        """

        self.ground_truth = ground_truth
        # Checks        
        weighted_predictions = np.array(weighted_predictions)
        self.weighted_predictions = weighted_predictions
        if weighted_predictions is not None:
            if list(np.unique(weighted_predictions)) == [0,1]:
                raise ValueError("Please provide a weighted sequence for weighted_predictions")
        
        binary_predictions = np.array(binary_predictions)
        self.binary_predictions = binary_predictions
        if binary_predictions is not None:
            if list(np.unique(binary_predictions)) == ([1] or [0]):
                raise ValueError("One class is not represented in your binary predictions")
            if list(np.unique(binary_predictions)) != [0,1]:
                raise ValueError("Please provide a binary sequence for binary_predictions")

        if weighted_predictions is not None:
            self.batched = True if weighted_predictions.ndim==2 else False
        if (binary_predictions is not None) and (self.batched is None):
            self.batched = True if binary_predictions.ndim==2 else False            
        self.n_samples = len(weighted_predictions) if self.batched else None
        self.bootstrap_stats = RCC_Scores()

        super().__init__(ground_truth=self.ground_truth, random_state=random_state)

    def __sensitivity(self, ground_truth, prediction):
        """
        Computes the sensitivity of the reconstructed binary networks.
        """
        
        # Scores & INFO: 
        # -----
        # tn => True negative
        # fp => False positive
        # fn => False negative
        # tp => True positive
        _, _, fn, tp = confusion_matrix(ground_truth, prediction).ravel() 
        sensitivity = tp / (tp + fn) if (tp + fn)>0 else 0
        return sensitivity
    
    def __specificity(self, ground_truth, prediction):
        """
        Computes the specificity of the reconstructed binary networks.
        """
        
        # Scores & INFO: 
        # -----
        # tn => True negative
        # fp => False positive
        # fn => False negative
        # tp => True positive
        
        tn, fp, _, _ = confusion_matrix(ground_truth, prediction).ravel() 
        specificity = tn / (tn + fp) if (tn + fp)>0 else 0
        return specificity
        
    def __positive_predictive_value(self, ground_truth, prediction):
        """
        Computes the positive predictive value of the reconstructed binary networks.
        """
        
        # Scores & INFO: 
        # -----
        # tn => True negative
        # fp => False positive
        # fn => False negative
        # tp => True positive
        _, fp, _, tp = confusion_matrix(ground_truth, prediction).ravel() 
        positive_predictive_value = tp / (tp + fp) if (tp + fp)>0 else 0
        return positive_predictive_value
    
    def __negative_predictive_value(self, ground_truth, prediction): 
        """
        Computes the negative predictive value of the reconstructed binary networks.
        """
        
        # Scores & INFO: 
        # -----
        # tn => True negative
        # fp => False positive
        # fn => False negative
        # tp => True positive
        tn, _, fn, _ = confusion_matrix(ground_truth, prediction).ravel() 
        negative_predictive_value = tn / (tn + fn) if (tn + fn)>0 else 0
        return  negative_predictive_value
        
    def compute_confusion_matrix(self):
        """
        Computes the sensitivity, the specificity and the predictive values of the reconstructed binary networks.
        """
        
        # Scores & INFO: 
        # -----
        # tn => True negative
        # fp => False positive
        # fn => False negative
        # tp => True positive
        
        if self.batched:
            sensitivity, specificity = np.zeros((self.n_samples,)), np.zeros((self.n_samples,))
            positive_predictive_value, negative_predictive_value = np.zeros((self.n_samples,)), np.zeros((self.n_samples,))            
            for i in range(self.n_samples):
                tn, fp, fn, tp = confusion_matrix(self.ground_truth, self.binary_predictions[i]).ravel() 
                sensitivity[i] = tp / (tp + fn) if (tp + fn)>0 else 0
                specificity[i] = tn / (tn + fp) if (tn + fp)>0 else 0
                positive_predictive_value[i] = tp / (tp + fp) if (tp + fp)>0 else 0
                negative_predictive_value[i] = tn / (tn + fn) if (tn + fn)>0 else 0
            return sensitivity, specificity, positive_predictive_value, negative_predictive_value
        else:
            tn, fp, fn, tp = confusion_matrix(self.ground_truth, self.binary_predictions).ravel() 
            sensitivity = tp / (tp + fn) if (tp + fn)>0 else 0
            specificity = tn / (tn + fp) if (tn + fp)>0 else 0
            positive_predictive_value = tp / (tp + fp) if (tp + fp)>0 else 0
            negative_predictive_value = tn / (tn + fn) if (tn + fn)>0 else 0
            return sensitivity, specificity, positive_predictive_value, negative_predictive_value

    """ def compute_auc(self):
        if self.batched:
            aucs = np.zeros((self.n_samples,))
            for i in range(self.n_samples):
                aucs[i] = roc_auc_score(self.ground_truth, self.weighted_predictions[i,:])
        else:
            return roc_auc_score(self.ground_truth, self.weighted_predictions) """
        
    def auc(self, n_bootstrap=None, level=95):
        if self.batched:
            # Real scores            
            aucs = np.zeros((self.n_samples,))
            for i in range(self.n_samples):
                aucs[i] = roc_auc_score(self.ground_truth, self.weighted_predictions[i])

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                auc_boot = self.bootstrap_dataset(
                    self.weighted_predictions, roc_auc_score, n_bootstrap=n_bootstrap
                )
        else:
            # Real scores
            aucs = roc_auc_score(self.ground_truth, self.weighted_predictions)

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                auc_boot = self.bootstrap_sample(
                    self.weighted_predictions, roc_auc_score, n_bootstrap=n_bootstrap
                )       
        # Bootstrapping the stats
        if n_bootstrap is not None:  
            self.bootstrap_stats.auc = self.compute_stats(RCC_Scores, auc_boot, level=95)
            self.bootstrap_stats.auc.true = aucs

        return aucs
        
    def sensitivity(self, n_bootstrap=1000):
        if self.batched:
            # Real scores            
            senses = np.zeros((self.n_samples,))
            for i in range(self.n_samples):
                senses[i] = self.__sensitivity(self.ground_truth, self.binary_predictions[i])

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                senses_boot = self.bootstrap_dataset(
                    self.binary_predictions, self.__sensitivity, n_bootstrap=n_bootstrap
            )
        else:
            # Real scores
            senses = self.__sensitivity(self.ground_truth, self.binary_predictions)

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                senses_boot = self.bootstrap_sample(
                self.binary_predictions, self.__sensitivity, n_bootstrap=n_bootstrap
            )      
        # Bootstrapping the stats
        if n_bootstrap is not None:  
            self.bootstrap_stats.sensitivity = self.compute_stats(RCC_Scores, senses_boot, level=95)
            self.bootstrap_stats.sensitivity.true = senses

        return senses
        
    def specificity(self, n_bootstrap=1000):
        if self.batched:
            # Real scores            
            specs = np.zeros((self.n_samples,))
            for i in range(self.n_samples):
                specs[i] = self.__specificity(self.ground_truth, self.binary_predictions[i])

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                specs_boot = self.bootstrap_dataset(
                self.binary_predictions, self.__specificity, n_bootstrap=n_bootstrap
            )
        else:
            # Real scores
            specs = self.__specificity(self.ground_truth, self.binary_predictions)

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                specs_boot = self.bootstrap_sample(
                self.binary_predictions, self.__specificity, n_bootstrap=n_bootstrap
            )    
        # Bootstrapping the stats
        if n_bootstrap is not None:  
            self.bootstrap_stats.specificity = self.compute_stats(RCC_Scores, specs_boot, level=95)
            self.bootstrap_stats.specificity.true = specs

        return specs
        
    def positive_predictive_value(self, n_bootstrap=1000):        
        if self.batched:
            # Real scores            
            ppvs = np.zeros((self.n_samples,))
            for i in range(self.n_samples):
                ppvs[i] = self.__positive_predictive_value(self.ground_truth, self.binary_predictions[i])

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                ppvs_boot = self.bootstrap_dataset(
                self.binary_predictions, self.__positive_predictive_value, n_bootstrap=n_bootstrap
            )
        else:
            # Real scores
            ppvs = self.__positive_predictive_value(self.ground_truth, self.binary_predictions)

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                ppvs_boot = self.bootstrap_sample(
                self.binary_predictions, self.__positive_predictive_value, n_bootstrap=n_bootstrap
            )
        # Bootstrapping the stats
        if n_bootstrap is not None:  
            self.bootstrap_stats.ppv = self.compute_stats(RCC_Scores, ppvs_boot, level=95)
            self.bootstrap_stats.ppv.true = ppvs

        return ppvs
        
    def negative_predicitive_value(self, n_bootstrap=1000):    
        if self.batched:
            # Real scores            
            npvs = np.zeros((self.n_samples,))
            for i in range(self.n_samples):
                npvs[i] = self.__negative_predictive_value(self.ground_truth, self.binary_predictions[i])

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                npvs_boot = self.bootstrap_dataset(
                self.binary_predictions, self.__negative_predictive_value, n_bootstrap=n_bootstrap
            )
        else:
            # Real scores
            npvs = self.__negative_predictive_value(self.ground_truth, self.binary_predictions)

            # Bootstrapping the stats
            if (n_bootstrap is not None):
                if n_bootstrap<500:
                    warnings.warn("Bootstrapping with less than 500 resamples is not encouraged")
                npvs_boot = self.bootstrap_sample(
                self.binary_predictions, self.__negative_predictive_value, n_bootstrap=n_bootstrap
            )
        # Bootstrapping the stats
        if n_bootstrap is not None:  
            self.bootstrap_stats.npv = self.compute_stats(RCC_Scores, npvs_boot, level=95)
            self.bootstrap_stats.npv.true = npvs

        return npvs
        
    def all(self, n_bootstrap=1000):
        pass