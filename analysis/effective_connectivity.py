import os
import numpy as  np
import networkx as nx

# Relative imports
from analysis.utils import process_subject_summary
from analysis.plotting import plot_evidence

class Subject_Effective_Connectivity():
    def __init__(self, directory, name_subject, ROI_Labels=None) -> None:
        # Paths
        self.name_subject = name_subject
        self.output_dir_subject = os.path.join(directory,name_subject)
        self.numerical = os.path.join(self.output_dir_subject,"Numerical")
        self.figures = os.path.join(self.output_dir_subject,"Figures")

        # ROI labels if present
        if ROI_Labels is not None:
            self.ROI_Labels = ROI_Labels
            self.labels = True
        else:
            self.labels = False

        # Process summary
        self.results_subject, self.header, ROI_CODE_Converter = process_subject_summary(
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
            lag_results = self.results_subject[file_ID][lag,:]
            edge, _0Indexed_edge = list(edge), list(_0Indexed_edge)
            # Create edge
            i2j_key, j2i_key, ij_key = self.__interaction_keys(*edge)
            if weighted and bidirectional:
                w = lag_results[self.header["Score "+i2j_key]]+lag_results[self.header["Score "+ij_key]]
                w_rev = lag_results[self.header["Score "+j2i_key]]+lag_results[self.header["Score "+ij_key]]
            elif (weighted) and (not bidirectional):
                w = lag_results[self.header["Score "+i2j_key]]
                w_rev = lag_results[self.header["Score "+j2i_key]]
            elif (not weighted) and (bidirectional):
                w = lag_results[self.header["Evidence "+i2j_key]]+lag_results[self.header["Evidence "+ij_key]]
                w_rev = lag_results[self.header["Evidence "+j2i_key]]+lag_results[self.header["Evidence "+ij_key]]
            else:                
                w = lag_results[self.header["Evidence "+i2j_key]]
                w_rev = lag_results[self.header["Evidence "+j2i_key]]
            # Add edges
            if not np.isnan(w):
                G_lag.add_edge(*edge, weight=w)
                network_lag[_0Indexed_edge[0],_0Indexed_edge[1]] = w
            else:
                network_lag[_0Indexed_edge[0],_0Indexed_edge[1]] = 0
            if not np.isnan(w_rev):
                G_lag.add_edge(*edge[::-1], weight=w_rev)
                network_lag[_0Indexed_edge[1],_0Indexed_edge[0]] = w_rev
            else:
                network_lag[_0Indexed_edge[1],_0Indexed_edge[0]] = 0
        return G_lag, network_lag

    def Lagged_Networks(self, weighted=False, bidirectional=False):
        Gs, networks = dict(), dict()
        for i, lag in enumerate(self.lags):
            Gs[lag], networks[lag] = self.__Network_AT_Lag(lag, weighted=weighted, bidirectional=bidirectional)
        return Gs, networks

    def plot_interaction_ij(self, roi_i, roi_j, **kwargs):
        roi_results = self.results_subject[self.__get_file_ID(roi_i, roi_j)]
        lags = roi_results[:, self.header["time-lags"]]
        i2j_key, j2i_key, ij_key = self.__interaction_keys(roi_i, roi_j)

        # Mean results
        mean_i2j = roi_results[:, self.header[i2j_key]]
        mean_j2i = roi_results[:, self.header[j2i_key]]
        mean_i2js = roi_results[:, self.header[i2j_key+"Surrogate"]]
        mean_j2is = roi_results[:, self.header[j2i_key+"Surrogate"]]

        # Standard Errors
        sem_i2j = roi_results[:, self.header["SEM "+i2j_key]]
        sem_j2i = roi_results[:, self.header["SEM "+j2i_key]]
        sem_i2js = roi_results[:, self.header["SEM "+i2j_key+"Surrogate"]]
        sem_j2is = roi_results[:, self.header["SEM "+j2i_key+"Surrogate"]]

        # Binary evidence
        evidence_i2j = roi_results[:, self.header["Evidence "+i2j_key]]
        evidence_j2i = roi_results[:, self.header["Evidence "+j2i_key]]
        evidence_ij = roi_results[:, self.header["Evidence "+ij_key]]

        # Naming(s)
        if 'save' in kwargs.keys():
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