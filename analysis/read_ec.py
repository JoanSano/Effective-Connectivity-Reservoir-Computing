import os
import numpy as  np
import networkx as nx
from tqdm import tqdm

# Relative imports
from analysis.utils import process_subject_summary, RCC_Scores, save_networks
from analysis.plotting import plot_evidence
from methods.utils import directionality_test_RCC

class Subject_EC():
    def __init__(self, directory, name_subject, ROI_Labels=None) -> None:
        # Paths
        self.name_subject = name_subject
        self.output_dir_subject = os.path.join(directory,name_subject)
        self.numerical = os.path.join(self.output_dir_subject,"Numerical")
        self.figures = os.path.join(self.output_dir_subject,"Figures")
        self.networks = os.path.join(self.output_dir_subject,"Networks")

        # ROI labels if present
        if ROI_Labels is not None:
            self.ROI_Labels = ROI_Labels
            self.labels = True
        else:
            self.labels = False

        # Process summary
        self.results_subject, self.headers, ROI_CODE_Converter = process_subject_summary(
            directory, name_subject
        )

        # Deal with ROIs and edge notation. IMPORTANT: ROIs are already sorted from the method itself.
        # Files with be created in ascending order
        # In contrast, edges might be reffered by it's index or by its number in the original dataset.
        # This needs to be dealt with carefully. 
        self._0IndexedNode_TO_ROI = ROI_CODE_Converter["0IndexedNode_TO_ROI"]
        self._0Indexed_TO_fileID_pairwise_keys = ROI_CODE_Converter["0Indexed_TO_fileID_pairwise_keys"]
        self._0Indexed_TO_Original_pariwise_keys = ROI_CODE_Converter["0Indexed_TO_Original_pariwise_keys"]
        self._0Indexed_ROIs = list(self._0IndexedNode_TO_ROI.keys())
        self.ROIs = list(self._0IndexedNode_TO_ROI.values())

        # Network and method properties
        self.N_rois, self.N_edges = self._0Indexed_ROIs[-1]+1, len(self.results_subject)
        self.lags =  self.results_subject[0][:,0]
        self.key2lag = dict(zip(self.lags, range(0,len(self.lags))))
    
    def __get_file_ID(self, roi_i, roi_j):
        roi_i, roi_j = sorted([roi_i, roi_j])
        try:
            _0indexed_pair = self._0Indexed_TO_Original_pariwise_keys[(roi_i,roi_j)]
            pair_ID = self._0Indexed_TO_fileID_pairwise_keys[_0indexed_pair]
            return pair_ID
        except:
            raise ValueError(f"No data for the pair of input ROIs ({roi_i},{roi_j})")
        
    def __interaction_keys(self, roi_i, roi_j):
        i2j_key = str(roi_i) + ' --> ' + str(roi_j)
        j2i_key = str(roi_j) + ' --> ' + str(roi_i)
        ij_key = str(roi_i) + ' <--> ' + str(roi_j)
        return i2j_key, j2i_key, ij_key

    def __Network_AT_Lag(self, lag, weighted=False, bidirectional=False):   
        # Creat adjacency matrix
        network_lag = np.zeros((self.N_rois,self.N_rois))
        # Create network
        G_lag = nx.DiGraph(directed=True)
        G_lag.add_nodes_from(self.ROIs)
        # Select the desired lag
        lag = self.key2lag[lag]
        # Iterate through 
        for edge, _0Indexed_edge in self._0Indexed_TO_Original_pariwise_keys.items():
            file_ID = self._0Indexed_TO_fileID_pairwise_keys[_0Indexed_edge]
            header_interaction = self.headers[file_ID]
            lag_results = self.results_subject[file_ID][lag,:]
            edge, _0Indexed_edge = list(edge), list(_0Indexed_edge)
            
            # Create edge
            i2j_key, j2i_key, ij_key = self.__interaction_keys(*edge)
            if weighted and bidirectional:
                w = lag_results[header_interaction["Score "+i2j_key]]+lag_results[header_interaction["Score "+ij_key]]
                w_rev = lag_results[header_interaction["Score "+j2i_key]]+lag_results[header_interaction["Score "+ij_key]]
            elif (weighted) and (not bidirectional):
                w = lag_results[header_interaction["Score "+i2j_key]]
                w_rev = lag_results[header_interaction["Score "+j2i_key]]
            elif (not weighted) and (bidirectional):
                w = lag_results[header_interaction["Evidence "+i2j_key]]+lag_results[header_interaction["Evidence "+ij_key]]
                w_rev = lag_results[header_interaction["Evidence "+j2i_key]]+lag_results[header_interaction["Evidence "+ij_key]]
            else:                
                w = lag_results[header_interaction["Evidence "+i2j_key]]
                w_rev = lag_results[header_interaction["Evidence "+j2i_key]]
            # Add edges
            if not np.isnan(w):
                G_lag.add_edge(*edge, weight=w)
            if not np.isnan(w_rev):
                G_lag.add_edge(*edge[::-1], weight=w_rev)
        return G_lag, nx.to_numpy_array(G_lag, nodelist=sorted(self.ROIs))
    
    def get_interaction_scores(self, roi_i, roi_j):
        pair_ID = self.__get_file_ID(roi_i, roi_j)
        roi_results = self.results_subject[pair_ID]
        header_interaction = self.headers[pair_ID]
        lags = roi_results[:, header_interaction["time-lags"]]
        i2j_key, j2i_key, ij_key = self.__interaction_keys(roi_i, roi_j)

        # Mean results
        mean = RCC_Scores()
        mean.i2j = roi_results[:, header_interaction[i2j_key]]
        mean.j2i = roi_results[:, header_interaction[j2i_key]]
        mean.i2js = roi_results[:, header_interaction[i2j_key+"Surrogate"]]
        mean.j2is = roi_results[:, header_interaction[j2i_key+"Surrogate"]]

        # Standard Errors
        sem = RCC_Scores()
        sem.i2j = roi_results[:, header_interaction["SEM "+i2j_key]]
        sem.j2i = roi_results[:, header_interaction["SEM "+j2i_key]]
        sem.i2js = roi_results[:, header_interaction["SEM "+i2j_key+"Surrogate"]]
        sem.j2is = roi_results[:, header_interaction["SEM "+j2i_key+"Surrogate"]]

        # Binary evidence
        evidence = RCC_Scores()
        evidence.i2j = roi_results[:, header_interaction["Evidence "+i2j_key]]
        evidence.j2i = roi_results[:, header_interaction["Evidence "+j2i_key]]
        evidence.ij = roi_results[:, header_interaction["Evidence "+ij_key]]
        return lags, mean, sem, evidence

    def Lagged_Networks(self, weighted=False, bidirectional=False):
        Gs, networks = dict(), dict()
        for i, lag in enumerate(self.lags):
            Gs[lag], networks[lag] = self.__Network_AT_Lag(lag, weighted=weighted, bidirectional=bidirectional)
        return Gs, networks
    
    def save_networks(self, networks: dict, directory=None, save_as="csv"):
        if directory is None:
            directory = self.networks
        if not os.path.exists(directory):
            os.mkdir(directory)
        save_networks(networks, directory, save_as=save_as)        

    def plot_interaction_ij(self, roi_i, roi_j, **kwargs):
        # Read results
        lags, mean, sem, evidence = self.get_interaction_scores(roi_i, roi_j)

        # Naming(s)
        if 'save' in kwargs.keys() and kwargs['save']:
            format = kwargs["format"] if 'format' in kwargs.keys() else 'png'
            dpi = kwargs["dpi"] if 'dpi' in kwargs.keys() else 300
            name_subject_RCC = self.name_subject + '_ROIs-' +str(roi_i) + 'vs' + str(roi_j)
            name_subject_RCC_figure = os.path.join(self.figures, name_subject_RCC+'.' + format)
        else:
            name_subject_RCC_figure = None
            dpi = None

        if self.labels:
            i2jlabel = self.ROI_Labels[roi_i] + r"$\rightarrow$" + self.ROI_Labels[roi_j]
            j2ilabel = self.ROI_Labels[roi_j] + r"$\rightarrow$" + self.ROI_Labels[roi_i]
            ijlabel = self.ROI_Labels[roi_i] + r"$\leftrightarrow$" + self.ROI_Labels[roi_j]
            name_roi_i, name_roi_j = self.ROI_Labels[roi_i], self.ROI_Labels[roi_j]
        else:
            i2jlabel = str(roi_i) + r"$\rightarrow$" + str(roi_j)
            j2ilabel = str(roi_j) + r"$\rightarrow$" + str(roi_i)
            ijlabel = str(roi_i) + r"$\leftrightarrow$" + str(roi_j)            
            name_roi_i, name_roi_j = str(roi_i), str(roi_j)

        # Label to refer to for the predictability measure
        score_label = kwargs["score_label"] if 'score_label' in kwargs.keys() else r"$\rho_{\tau}$"

        # Y limits
        ylims = kwargs["ylims"] if 'ylims' in kwargs.keys() else None
        
        # Plotting
        x_label = kwargs["x_label"] if 'x_label' in kwargs.keys() else r"$\tau$"+"(steps)"
        title = kwargs["title"] if 'title' in kwargs.keys() else None
        plot_evidence(
            lags,
            {"data": mean.i2j, "error": sem.i2j, "label": score_label+f"({name_roi_i},{name_roi_j})", "color": "darkorange", "style": "-", "linewidth": 1, "alpha": 1}, 
            {"data": mean.j2i, "error": sem.j2i, "label": score_label+f"({name_roi_j},{name_roi_i})", "color": "green", "style": "-", "linewidth": 1, "alpha": 1}, 
            {"data": mean.i2js, "error": sem.i2js, "label": score_label+f"({name_roi_i},{name_roi_j}"+r"$_{S}$"+")", "color": "bisque", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
            {"data": mean.j2is, "error": sem.j2is, "label": score_label+f"({name_roi_j},{name_roi_i}"+r"$_{S}$"+")", "color": "lightgreen", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
            title=title , y_label="Eff. connectivity scoring", x_label=x_label, limits=ylims, #scale=0.720, 
            significance_marks=[
                {"data": evidence.i2j, "color": "blue", "label": i2jlabel},
                {"data": evidence.j2i, "color": "red", "label": j2ilabel},
                {"data": evidence.ij, "color": "purple", "label": ijlabel}
            ],
            save=name_subject_RCC_figure, dpi=dpi
        ) 

class Group_EC():
    def __init__(self, directory, ROI_Labels=None) -> None:
        self.directory = directory
        self.subjects = dict()
        ss = [s for s in os.listdir(directory) if s.split(".")[-1] not in ['txt','csv','tsv','json','png','svg','jpg','jpeg']]
        for i, s in enumerate(ss):
            self.subjects[s] = i
        self.Ns = len(self.subjects)

        # ROI labels if present
        self.ROI_Labels = ROI_Labels
        if ROI_Labels is not None:
            self.labels = True
        else:
            self.labels = False

    def get_networks_Adjacency(self, weighted=False, bidirectional=False, save_in=False):
        group_matrices = dict()
        for s, i in self.subjects.items():
            # Load networks
            SEC = Subject_EC(self.directory, s, ROI_Labels=self.ROI_Labels) 
            Gs, matrices = SEC.Lagged_Networks(weighted=weighted, bidirectional=bidirectional)
            
            # Save networks
            if save_in==True:
                SEC.save_networks(matrices)

            # Compact them
            for lag in Gs.keys():
                if i == 0: # First subject of the dataset
                    group_matrices[lag] = np.zeros((self.Ns, SEC.N_rois, SEC.N_rois))
                group_matrices[lag][i,...] = matrices[lag]
            del SEC                
        return group_matrices
    
    def save_networks(self, networks: dict, directory=None, save_as="csv"):
        if directory is None:
            directory = self.directory
        if not os.path.exists(directory):
            os.mkdir(directory)
        
        code = {v:k for k,v in self.subjects.items()}
        save_networks(networks, directory, save_as=save_as, group_names=code)

    def group_consistency(self, lag, threshold=0.7, make_nan=True, make_binary=False):
        nets = self.get_networks_Adjacency(weighted=False, bidirectional=False)[lag]
        if (np.unique(nets)!=np.array([0,1])).all():
            raise ValueError("Networks are not binary, consistency not defined")
        nets = nets.mean(axis=0)
        # Node order is already correct due to Subject_Effective_Connectivity function __Network_At_Lag()
        if make_binary:
            return nx.from_numpy_array(np.where(nets>=threshold, 1, 0), create_using=nx.DiGraph), np.where(nets>=threshold, 1, np.nan) if make_nan else np.where(nets>=threshold, 1, 0)  
        else:   
            return nx.from_numpy_array(np.where(nets>=threshold, nets, 0), create_using=nx.DiGraph), np.where(nets>=threshold, nets, np.nan) if make_nan else np.where(nets>=threshold, nets, 0) 

    def get_interaction_scores(self, roi_i, roi_j):
        mean_i2j, mean_j2i = [], []
        mean_i2js, mean_j2is = [], []
        # Create database of results
        for s, i in tqdm(self.subjects.items()):
            SEC = Subject_EC(self.directory, s, ROI_Labels=self.ROI_Labels)
            lags, mean, _, _ = SEC.get_interaction_scores(roi_i, roi_j)
            mean_i2j.append(mean.i2j), mean_j2i.append(mean.j2i)
            mean_i2js.append(mean.i2js), mean_j2is.append(mean.j2is)
        mean_i2j, mean_j2i = np.array(mean_i2j), np.array(mean_j2i)
        mean_i2js, mean_j2is = np.array(mean_i2js), np.array(mean_j2is)

        # Compact results
        mean = RCC_Scores()
        mean.i2j, mean.j2i, mean.i2js, mean.j2is = mean_i2j, mean_j2i, mean_i2js, mean_j2is
        return lags, mean
    
    def plot_interaction_ij(self, roi_i, roi_j, significance=0.05, permutations=False, bonferroni=True, **kwargs):
        # Read results
        lags, mean = self.get_interaction_scores(roi_i, roi_j)
        # Generate stats
        mean_i2j = mean.i2j.mean(axis=0)
        mean_j2i = mean.j2i.mean(axis=0)
        mean_i2js = mean.i2js.mean(axis=0)
        mean_j2is = mean.j2is.mean(axis=0)
        sem_i2j = mean.i2j.std(axis=0)/np.sqrt(self.Ns)
        sem_j2i = mean.j2i.std(axis=0)/np.sqrt(self.Ns)
        sem_i2js = mean.i2js.std(axis=0)/np.sqrt(self.Ns)
        sem_j2is = mean.j2is.std(axis=0)/np.sqrt(self.Ns)
        evidence_ij, evidence_i2j, evidence_j2i, AAA, BBB, CCC = directionality_test_RCC(
            mean.i2j.T, mean.j2i.T, mean.i2js.T, mean.j2is.T, lags=lags, 
            significance=significance, permutations=permutations, bonferroni=bonferroni
        )

        # Naming(s)
        if 'save' in kwargs.keys() and kwargs['save']:
            format = kwargs["format"] if 'format' in kwargs.keys() else 'png'
            dpi = kwargs["dpi"] if 'dpi' in kwargs.keys() else 300
            name_subject_RCC = self.name_subject + '_ROIs-' +str(roi_i) + 'vs' + str(roi_j)
            name_subject_RCC_figure = os.path.join(self.figures, name_subject_RCC+'.' + format)
        else:
            name_subject_RCC_figure = None
            dpi = None

        if self.labels:
            i2jlabel = self.ROI_Labels[roi_i] + r"$\rightarrow$" + self.ROI_Labels[roi_j]
            j2ilabel = self.ROI_Labels[roi_j] + r"$\rightarrow$" + self.ROI_Labels[roi_i]
            ijlabel = self.ROI_Labels[roi_i] + r"$\leftrightarrow$" + self.ROI_Labels[roi_j]
            name_roi_i, name_roi_j = self.ROI_Labels[roi_i], self.ROI_Labels[roi_j]
        else:
            i2jlabel = str(roi_i) + r"$\rightarrow$" + str(roi_j)
            j2ilabel = str(roi_j) + r"$\rightarrow$" + str(roi_i)
            ijlabel = str(roi_i) + r"$\leftrightarrow$" + str(roi_j)            
            name_roi_i, name_roi_j = str(roi_i), str(roi_j)

        # Label to refer to for the predictability measure
        score_label = kwargs["score_label"] if 'score_label' in kwargs.keys() else r"$\rho_{\tau}$"

        # Y limits
        ylims = kwargs["ylims"] if 'ylims' in kwargs.keys() else None
        
        # Plotting
        x_label = kwargs["x_label"] if 'x_label' in kwargs.keys() else r"$\tau$"+"(steps)"
        title = kwargs["title"] if 'title' in kwargs.keys() else None
        plot_evidence(
            lags,
            {"data": mean_i2j, "error": sem_i2j, "label": score_label+f"({name_roi_i},{name_roi_j})", "color": "darkorange", "style": "-", "linewidth": 1, "alpha": 1}, 
            {"data": mean_j2i, "error": sem_j2i, "label": score_label+f"({name_roi_j},{name_roi_i})", "color": "green", "style": "-", "linewidth": 1, "alpha": 1}, 
            {"data": mean_i2js, "error": sem_i2js, "label": score_label+f"({name_roi_i},{name_roi_j}"+r"$_{S}$"+")", "color": "bisque", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
            {"data": mean_j2is, "error": sem_j2is, "label": score_label+f"({name_roi_j},{name_roi_i}"+r"$_{S}$"+")", "color": "lightgreen", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
            title=title , y_label="Eff. connectivity scoring", x_label=x_label, limits=ylims, #scale=0.720, 
            significance_marks=[
                {"data": evidence_i2j, "color": "blue", "label": i2jlabel},
                {"data": evidence_j2i, "color": "red", "label": j2ilabel},
                {"data": evidence_ij, "color": "purple", "label": ijlabel}
            ],
            save=name_subject_RCC_figure, dpi=dpi
        ) 

if __name__ == '__main__':
    pass