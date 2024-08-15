import argparse
import os
import glob
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Relative imports
from analysis_utils import *

""" # Recovering arguments from RCC reconstruction
parser_RCC = argparse.ArgumentParser()
opts_RCC = parser_RCC.parse_args()
with open("Results-RCC_arguments.txt", 'r') as f:
    opts_RCC.__dict__ = json.load(f)

# Recovering arguments from GC reconstruction
parser_GC = argparse.ArgumentParser()
opts_GC = parser_GC.parse_args()
with open("Results-GC_arguments.txt", 'r') as f:
    opts_GC.__dict__ = json.load(f) """

'''
column name                |||  Index value
===========================================
time-lags                   0
PPV                         1
NPV                         2
AUC                         3
SEM PPV                     4
SEM NPV                     5
SEM AUC                     6
'''

parser = argparse.ArgumentParser(f"\nPopulation analysis of Reservoir Computing Causality.\nOf course it needs a folder with the results of each subject in the population.")
parser.add_argument('-f', '--figs', type=int, default=[1,2,3], nargs='+', help="Figures to plot - 0 to plot all")
opts = parser.parse_args()

def relative_increase(lags, x, y=None, max_lag=10, include_positive=False, delta_x=None, delta_y=None):
    """
    Computes the relative increase and the associated Standard Error of the Mean (SEM) for the secified number of lags.
    The increase is measured as x over y; hence, the reference value is y. If y is None, it is understood that x is greater than 0.
    Make sure that lags, x, and y have the same lengths.
    """
    lags = np.array(lags)

    if include_positive:
        mask = (np.abs(lags)<=max_lag) * (lags!=0)
    else:
        mask = (lags>=-max_lag) * (lags<0)
    
    lags = lags[mask]
    x = x[mask]
    delta_x = x*0 if delta_x is None else np.abs(delta_x[mask])
    if y is not None:
        y = y[mask]         
        delta_y = y*0 if delta_y is None else np.abs(delta_y[mask])
        rel_increase = 100 * (x-y)/y
        error = 100 * ( np.abs(1/y)*delta_x + np.abs(x/(y*y))*delta_y )
    else:
        rel_increase = x
        error = delta_x
    
    # One-sided Welch's t-test - Manual implementation
    from scipy.stats import t
    t_scores = (x-y)/np.sqrt(delta_x**2+delta_y**2)
    freedom = list(np.array((delta_x**2+delta_y**2)**2 / ( (delta_x**4/99) + (delta_y**4/99) ), dtype = np.int16))
    p_vals = 1-t.cdf(t_scores, freedom)
    significance = []
    for p_v in p_vals:
        if p_v > 0.05:
            significance.append("")
        elif (p_v<=0.05) and (p_v>=0.01):
            significance.append("*")
        elif (p_v<=0.01) and (p_v>=0.001):
            significance.append("**")
        else:
            significance.append("***")
    return lags, rel_increase, error, significance

if __name__ == '__main__':
    lags_to_see_show_GC = False
    folder = os.path.join(os.getcwd(),"Results-Figures_Global-Analysis")
    if not os.path.exists(folder):
        os.mkdir(folder) 

    ####################
    ### Load results ###
    ####################
    files = glob.glob(f"Results-Metrics_Method-*")
    sims = np.unique(np.array([int(f.split("_")[2].split("-")[1]) for f in files])) # Automatically sorted
    lengths = np.unique(np.array([int(f.split("_")[-1].split("-")[-1].split(".")[0]) for f in files]))
    results_RCC = np.empty(shape=len(sims), dtype=object)
    results_GC = np.empty(shape=len(sims), dtype=object)
    """
    - Data is structures the following way:
    results_method = [sims][lengths] contains the information according to the previous help box (see up)
    Both the length and the simulations are sorted with the numpy method.
    """

    if lags_to_see_show_GC:
        lags_to_see = [-4,-3,-2,-1] # for GC results 
        index = np.argwhere(sims==15)
        sims = np.delete(sims, index)
    else:
        lags_to_see = [-2,-1,1,2] # for RCC results

    length_convergence = np.zeros((len(lags_to_see), len(sims), len(lengths)))
    length_convergence_SEM = np.zeros((len(lags_to_see), len(sims), len(lengths)))
    pearson_rho_all_tau = []

    for s, sim in enumerate(sims):
        results_sim_RCC = np.empty(shape=len(lengths), dtype=object)
        results_sim_GC = np.empty(shape=len(lengths), dtype=object)
        length_performance = []
        for i, L in enumerate(lengths):
            # Reservoir
            f = f"Results-Metrics_Method-RCC_Sim-{sim}_Length-{L}.tsv"
            if os.path.exists(f):
                results_sim_RCC[i] = np.genfromtxt(f, delimiter="\t", skip_header=1)
            else:
                results_sim_RCC[i] = None
            
            # Granger 
            f = f"Results-Metrics_Method-GC_Sim-{sim}_Length-{L}.tsv"
            if os.path.exists(f):
                results_sim_GC[i] = np.genfromtxt(f, delimiter="\t", skip_header=1)
            else:
                results_sim_GC[i] = None

            # Length convergence
            if lags_to_see_show_GC:
                for l, tau in enumerate(lags_to_see):   
                    id = np.argmax(np.where(results_sim_GC[i][:,0]==tau, 1, 0))
                    length_convergence[l,s,i] = results_sim_GC[i][id,3]
                    length_convergence_SEM[l,s,i] = results_sim_GC[i][id,6]
                length_performance.append(results_sim_GC[i][:,3])
                lags = results_sim_GC[i][:,0]
            else:
                for l, tau in enumerate(lags_to_see):
                    id = np.argmax(np.where(results_sim_RCC[i][:,0]==tau, 1, 0))
                    length_convergence[l,s,i] = results_sim_RCC[i][id,3]
                    length_convergence_SEM[l,s,i] = results_sim_RCC[i][id,6]
                length_performance.append(results_sim_RCC[i][:,3])
                lags = results_sim_RCC[i][:,0]
        
        # Length convergence for all lags tested for each simulation
        length_performance = np.array(length_performance)
        pearson_rho_all_tau.append(np.array([
            pearsonr(lengths, length_performance[:,t]) for t in range(length_performance.shape[-1])
        ]))
        results_RCC[s] = results_sim_RCC
        results_GC[s] = results_sim_GC

    ###############
    ### Figures ###
    ###############
    ### Figure of length convergence for short lags
    limits = [(0,1), (0,1), (0,1), (0,1), (0,1)]
    colors = ["gold", "red", "darkorange", "green", "blue", "darkviolet", "black"]
    fmt = "png"
    dpi = 500
    figs_to_do = opts.figs 

    if (1 or 0) in figs_to_do:
        fig, ax = plt.subplots(figsize=(10,4))
        ax.remove()
        left, bottom, width, height = [0.06, 0.12 , 0.2, 0.8]
        ax1 = fig.add_axes([left, bottom, width, height])
        ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)
        ax1.set_yticks([0.2,0.5,1])
        ax1.set_xticks(range(1,len(sims)+1)), ax1.set_xticklabels(sims)
        ax1.set_xlim([0.5,len(sims)+0.5]), ax1.set_ylim([0.2,1])
        ax1.set_xlabel("Simulation (#)", fontsize=15), ax1.set_ylabel("AUC", fontsize=15, labelpad=0)
        ax1.hlines(y=0.5, xmin=0, xmax=len(sims)+0.2, color="lightgray", linestyle='dashed', linewidth=1)
        ax1.text(1.1, 0.98, r'$\tau=$'+f"{lags_to_see[0]}", fontsize=20, fontweight="bold")
        for i, L in enumerate(lengths):
            x = [r-0.25+i*0.5/len(lengths) for r in range(1,len(sims)+1)]
            ax1.scatter(x, length_convergence[0,:,i], s=70, marker='o', color=colors[i], label=f"L={L}%")
            #ax1.errorbar(x, length_convergence[0,:,i], yerr=length_convergence_SEM[0,:,i], color=colors[i], fmt='none')
        for s, _ in enumerate(sims):
            reg = LinearRegression().fit(np.array(lengths).reshape(-1,1), length_convergence[0,s,:])
            pred = reg.predict(np.linspace(lengths[0], lengths[-1]).reshape(-1,1))
            x = np.linspace(s+1-0.4,s+1+0.4,len(pred))
            ax1.plot(x, pred, label=None, color="gray")
            pr = pearsonr(lengths, length_convergence[0,s,:])
            if pr[1] <= 0.05:
                ax1.text(s+.5, 0.225, r'$\rho=$'+f"{round(pr[0],2)}", fontsize=12, rotation=60)  
        ax1.legend(frameon=False, ncols=1, loc="upper right", bbox_to_anchor=(1.1, 0.57, 0.2, 0.5))

        left, bottom, width, height = [0.32, 0.12 , 0.2, 0.8]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.spines["top"].set_visible(False), ax2.spines["right"].set_visible(False), ax2.spines["left"].set_visible(False)
        ax2.set_yticks([])
        ax2.set_xticks(range(1,len(sims)+1)), ax2.set_xticklabels(sims)
        ax2.set_xlim([0.5,len(sims)+0.5]), ax2.set_ylim([0.2,1])
        ax2.set_xlabel("Simulation (#)", fontsize=15)
        ax2.hlines(y=0.5, xmin=0, xmax=len(sims)+0.2, color="lightgray", linestyle='dashed', linewidth=1)
        ax2.text(1.1, 0.98, r'$\tau=$'+f"{lags_to_see[1]}", fontsize=20, fontweight="bold")
        for i, L in enumerate(lengths):
            x = [r-0.25+i*0.5/len(lengths) for r in range(1,len(sims)+1)]
            ax2.scatter(x, length_convergence[1,:,i], s=70, marker='o', color=colors[i])
        for s, _ in enumerate(sims):
            reg = LinearRegression().fit(np.array(lengths).reshape(-1,1), length_convergence[1,s,:])
            pred = reg.predict(np.linspace(lengths[0], lengths[-1]).reshape(-1,1))
            x = np.linspace(s+1-0.4,s+1+0.4,len(pred))
            ax2.plot(x, pred, label=None, color="gray")
            pr = pearsonr(lengths, length_convergence[1,s,:])
            if pr[1] <= 0.05:
                ax2.text(s+.5, 0.225, r'$\rho=$'+f"{round(pr[0],2)}", fontsize=12, rotation=60)  

        left, bottom, width, height = [0.55, 0.12 , 0.2, 0.8]
        ax3 = fig.add_axes([left, bottom, width, height])
        ax3.spines["top"].set_visible(False), ax3.spines["right"].set_visible(False), ax3.spines["left"].set_visible(False)
        ax3.set_yticks([])
        ax3.set_xticks(range(1,len(sims)+1)), ax3.set_xticklabels(sims)
        ax3.set_xlim([0.5,len(sims)+0.5]), ax3.set_ylim([0.2,1])
        ax3.set_xlabel("Simulation (#)", fontsize=15)
        ax3.hlines(y=0.5, xmin=0, xmax=len(sims)+0.2, color="lightgray", linestyle='dashed', linewidth=1)
        ax3.text(1.1, 0.98, r'$\tau=$'+f"{lags_to_see[2]}", fontsize=20, fontweight="bold")
        for i, L in enumerate(lengths):
            x = [r-0.25+i*0.5/len(lengths) for r in range(1,len(sims)+1)]
            ax3.scatter(x, length_convergence[2,:,i], s=70, marker='o', color=colors[i])
        for s, _ in enumerate(sims):
            reg = LinearRegression().fit(np.array(lengths).reshape(-1,1), length_convergence[2,s,:])
            pred = reg.predict(np.linspace(lengths[0], lengths[-1]).reshape(-1,1))
            x = np.linspace(s+1-0.4,s+1+0.4,len(pred))
            ax3.plot(x, pred, label=None, color="gray")
            pr = pearsonr(lengths, length_convergence[2,s,:])
            if pr[1] <= 0.05:
                ax3.text(s+.5, 0.225, r'$\rho=$'+f"{round(pr[0],2)}", fontsize=12, rotation=60)  

        left, bottom, width, height = [0.78, 0.12 , 0.2, 0.8]
        ax4 = fig.add_axes([left, bottom, width, height])
        ax4.spines["top"].set_visible(False), ax4.spines["right"].set_visible(False), ax4.spines["left"].set_visible(False)
        ax4.set_yticks([])
        ax4.set_xticks(range(1,len(sims)+1)), ax4.set_xticklabels(sims)
        ax4.set_xlim([0.5,len(sims)+0.5]), ax4.set_ylim([0.2,1])
        ax4.set_xlabel("Simulation (#)", fontsize=15)
        ax4.hlines(y=0.5, xmin=0, xmax=len(sims)+0.2, color="lightgray", linestyle='dashed', linewidth=1)
        ax4.text(1.1, 0.98, r'$\tau=$'+f"{lags_to_see[3]}", fontsize=20, fontweight="bold")
        for i, L in enumerate(lengths):
            x = [r-0.25+i*0.5/len(lengths) for r in range(1,len(sims)+1)]
            ax4.scatter(x, length_convergence[3,:,i], s=70, marker='o', color=colors[i])
        for s, _ in enumerate(sims):
            reg = LinearRegression().fit(np.array(lengths).reshape(-1,1), length_convergence[3,s,:])
            pred = reg.predict(np.linspace(lengths[0], lengths[-1]).reshape(-1,1))
            x = np.linspace(s+1-0.4,s+1+0.4,len(pred))
            ax4.plot(x, pred, label=None, color="gray")
            pr = pearsonr(lengths, length_convergence[3,s,:])
            if pr[1] <= 0.05:
                ax4.text(s+.5, 0.225, r'$\rho=$'+f"{round(pr[0],2)}", fontsize=12, rotation=60)  
        if lags_to_see_show_GC:
            plt.savefig(folder+"/Method-GC_Length-Convergence_Selected-Tau."+fmt, dpi=dpi, format=fmt)
        else:
            plt.savefig(folder+"/Method-RCC_Length-Convergence_Selected-Tau."+fmt, dpi=dpi, format=fmt)

    ### Figure of length convergence for all lags
    if (2 or 0) in figs_to_do:
        from analysis_utils import plot_RCC_Evidence
        colors = ["gold", "red", "green", "blue", "black"]
        rho_to_plot, pval_to_plot = [], []
        for s, sim in enumerate(sims):
            if sim != 7: # TODO: Incorporate different lengths!
                tp = pearson_rho_all_tau[s][:,0] * np.where(pearson_rho_all_tau[s][:,1]<=0.05, 1, np.nan) * np.where(pearson_rho_all_tau[s][:,0]>0, 1, np.nan) 
                rho_to_plot.extend([
                {"data": tp, "error": tp*0, "label": f"Sim {sim}%", "color": colors[s], "style": 'solid', "dots": True, "linewidth": 15, "alpha": 0.75},  
                ])
                #pval_to_plot.extend([
                #    {"data": np.where(pearson_rho_all_tau[s][:,1]<=0.05, 1, np.nan), "color": colors[s], "label": None}
                #])
            else: # Personalize based on the simulations run
                pads = (len(lags)-len(pearson_rho_all_tau[s][:,0]))//2
                series = np.pad(pearson_rho_all_tau[s][:,0], (pads,pads+1), 'constant', constant_values=(np.nan, np.nan))
                series_pval = np.pad(pearson_rho_all_tau[s][:,1], (pads,pads+1), 'constant', constant_values=(np.nan, np.nan))
                tp = series * np.where(series_pval<=0.05, 1, np.nan) * np.where(series>0, 1, np.nan) 
                rho_to_plot.extend([
                {"data": tp, "error": tp*0, "label": f"Sim {sim}%", "color": colors[s], "style": 'solid', "dots": True, "linewidth": 15, "alpha": 0.75},  
                ])
                #pval_to_plot.extend([G
                #    {"data": np.where(series_pval<=0.05, 1, np.nan), "color": colors[s], "label": None}
                #])
        plot_RCC_Evidence(
            np.where(lags!=0, lags, np.nan),
            *rho_to_plot,
            save=folder+"/Method-RCC_Length-Convergence_All-Tau."+fmt, 
            dpi=dpi, y_label=r"$\rho$(L,AUC)", x_label=r"$\tau$"+"(step)", limits=(0.7,1)
        )

    ### Explicit figure of RCC vs GC
    colors = ["red", "green", "blue"]
    if (3 or 0) in figs_to_do:
        fig, ax = plt.subplots(figsize=(10,20))
        ax.remove()
        
        ## SIM 1
        plt.gcf().text(0.41, 0.98, "Sim 1", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.825 , 0.425, 0.15]
        ax1 = fig.add_axes([left, bottom, width, height])
        ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)    
        yticks, space = [], 0.15
        for pp in range(1,4):
            labelRCC = "RCC" if pp==3 else None
            labelGC = "GC" if pp==3 else None
            ax1.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[0][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1], label=labelRCC)
            ax1.plot(results_GC[0][-1][:,0], results_GC[0][-1][:,pp]-space*(1-pp), linestyle='dashed', linewidth=2, color=colors[pp-1], label=labelGC)
            ax1.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax1.set_yticks(yticks), ax1.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax1.legend(frameon=True, fontsize=17, ncols=2, bbox_to_anchor=(0.55, 0.675, 0.2, 0.5))
        ax1.get_legend().legend_handles[0].set_color('black'), ax1.get_legend().legend_handles[1].set_color('black')
        ax1.set_xlabel(r"$\tau$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.825 , 0.425, 0.15]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.spines["top"].set_visible(False), ax2.spines["right"].set_visible(False)    
        labels, max_lag, space_x = ["PPV", "NPV", "AUC"], 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[lags<0], results_RCC[0][-1][:,pp][lags<0], y=results_GC[0][-1][:,pp], max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[0][-1][:,pp+3][lags<0], delta_y=results_GC[0][-1][:,pp+3]
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax2.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], label=labels[pp-1], markersize=10, alpha=0.5)
            ax2.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                ax2.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax2.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax2.set_xticks(cut_lag), ax2.set_xticklabels(cut_lag), ax2.set_ylim([-10,100])
        ax2.legend(frameon=True, fontsize=15, ncols=3, bbox_to_anchor=(0.85, 0.675, 0.2, 0.5))
        ax2.set_xlabel(r"$\tau$"+"(step)", fontsize=15), ax2.set_ylabel("Relative Increase (%)", fontsize=15)

        ## SIM 7
        plt.gcf().text(0.45, 0.78, "Sim 7", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.625 , 0.425, 0.15]
        ax3 = fig.add_axes([left, bottom, width, height])
        ax3.spines["top"].set_visible(False), ax3.spines["right"].set_visible(False)
        yticks, space = [], 0.5
        for pp in range(1,4):
            ax3.plot(np.where(results_RCC[1][-1][:,0]<0, results_RCC[1][-1][:,0], np.nan), np.where(results_RCC[1][-1][:,0]<0, results_RCC[1][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax3.plot(results_GC[1][-1][:,0], results_GC[1][-1][:,pp]-space*(1-pp), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax3.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax3.set_yticks(yticks), ax3.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=75, va="center", fontsize=15)
        ax3.set_xlabel(r"$\tau$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.625 , 0.425, 0.15]
        ax4 = fig.add_axes([left, bottom, width, height])
        ax4.spines["top"].set_visible(False), ax4.spines["right"].set_visible(False)    
        max_lag, space_x = 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                results_RCC[1][-1][:,0][results_RCC[1][-1][:,0]<0], results_RCC[1][-1][:,pp][results_RCC[1][-1][:,0]<0], y=results_GC[1][-1][-10:,pp], max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[1][-1][:,pp+3][results_RCC[1][-1][:,0]<0], delta_y=results_GC[1][-1][-10:,pp+3]
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax4.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], markersize=10, alpha=0.5)
            ax4.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                ax4.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax4.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax4.set_xticks(cut_lag), ax4.set_xticklabels(cut_lag), ax4.set_ylim([-10,90])
        ax4.set_xlabel(r"$\tau$"+"(step)", fontsize=15), ax4.set_ylabel("Relative Increase (%)", fontsize=15)
        
        ## SIM 15
        plt.gcf().text(0.45, 0.58, "Sim 15", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.425 , 0.425, 0.15]
        ax5 = fig.add_axes([left, bottom, width, height])
        ax5.spines["top"].set_visible(False), ax5.spines["right"].set_visible(False)
        yticks, space = [], 0.2
        for pp in range(1,4):
            ax5.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[2][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax5.plot(results_GC[2][-1][:,0], results_GC[2][-1][:,pp]-space*(1-pp), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax5.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax5.set_yticks(yticks), ax5.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax5.set_xlabel(r"$\tau$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.425 , 0.425, 0.15]
        ax6 = fig.add_axes([left, bottom, width, height])
        ax6.spines["top"].set_visible(False), ax6.spines["right"].set_visible(False)    
        max_lag, space_x = 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[lags<0], results_RCC[2][-1][:,pp][lags<0], y=results_GC[2][-1][:,pp], max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[2][-1][:,pp+3][lags<0], delta_y=results_GC[2][-1][:,pp+3]
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax6.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], markersize=10, alpha=0.5)
            ax6.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                ax6.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax6.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax6.set_xticks(cut_lag), ax6.set_xticklabels(cut_lag), ax6.set_ylim([-30,90])
        ax6.set_xlabel(r"$\tau$"+"(step)", fontsize=15), ax6.set_ylabel("Relative Increase (%)", fontsize=15)
        
        ## SIM 19
        plt.gcf().text(0.45, 0.38, "Sim 19", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.225 , 0.425, 0.15]
        ax7 = fig.add_axes([left, bottom, width, height])
        ax7.spines["top"].set_visible(False), ax7.spines["right"].set_visible(False)
        yticks, space = [], 0.7
        for pp in range(1,4):
            ax7.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[3][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax7.plot(results_GC[3][-1][:,0], results_GC[3][-1][:,pp]-space*(1-pp), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax7.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax7.set_yticks(yticks), ax7.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax7.set_xlabel(r"$\tau$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.225 , 0.425, 0.15]
        ax8 = fig.add_axes([left, bottom, width, height])
        ax8.spines["top"].set_visible(False), ax8.spines["right"].set_visible(False)    
        max_lag, space_x = 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[lags<0], results_RCC[3][-1][:,pp][lags<0], y=results_GC[3][-1][1:,pp], max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[3][-1][:,pp+3][lags<0], delta_y=results_GC[3][-1][1:,pp+3]
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax8.plot(h_data, np.where(rel_increase<200,rel_increase,np.nan), 'o-', linewidth=0.8, color=colors[pp-1], markersize=10, alpha=0.5)
            ax8.errorbar(h_data, rel_increase, yerr=np.where(error<100,error,np.nan), ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                if sm != (len(significance)-1):
                    ax8.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax8.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax8.set_xticks(cut_lag), ax8.set_xticklabels(cut_lag), ax8.set_ylim([-40,130])
        ax8.set_xlabel(r"$\tau$"+"(step)", fontsize=15), ax8.set_ylabel("Relative Increase (%)", fontsize=15)
        
        ## SIM 28
        plt.gcf().text(0.45, 0.19, "Sim 28", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.025 , 0.425, 0.15]
        ax9 = fig.add_axes([left, bottom, width, height])
        ax9.spines["top"].set_visible(False), ax9.spines["right"].set_visible(False)
        yticks, space = [], 0.2
        for pp in range(1,4):
            ax9.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[4][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax9.plot(results_GC[4][-1][:,0], results_GC[4][-1][:,pp]-space*(1-pp), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax9.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax9.set_yticks(yticks), ax9.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax9.set_xlabel(r"$\tau$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.025 , 0.4, 0.15]
        ax9 = fig.add_axes([left, bottom, width, height])
        ax9.spines["top"].set_visible(False), ax9.spines["right"].set_visible(False)  
        ax_bis = ax9.twinx()
        ax_bis.spines["top"].set_visible(False), ax_bis.spines['right'].set_color('red')
        labels, max_lag, space_x = ["PPV", "NPV", "AUC"], 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[(lags<0)*(lags>=-20)], results_RCC[4][-1][:,pp][(lags<0)*(lags>=-20)], y=results_GC[4][-1][:,pp], max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[4][-1][:,pp+3][(lags<0)*(lags>=-20)], delta_y=results_GC[4][-1][:,pp+3]
            )
            h_data = cut_lag + (2 - pp) * space_x
            if pp == 1:
                ax_bis.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], label=labels[pp-1], markersize=10, alpha=0.5)
                ax_bis.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
                for sm, s_mark in enumerate(significance):
                    ax_bis.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 50, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
            else:
                ax9.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], label=labels[pp-1], markersize=10, alpha=0.5)
                ax9.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
                for sm, s_mark in enumerate(significance):
                    ax9.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax9.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax9.set_xticks(cut_lag), ax9.set_xticklabels(cut_lag), ax9.set_ylim([-5,60])
        ax9.set_xlabel(r"$\tau$"+"(step)", fontsize=15), ax9.set_ylabel("Relative Increase (%)", fontsize=15)
        ax_bis.set_ylim([-1000,400]), ax_bis.set_yticks([0,100,200]), ax_bis.set_yticklabels([0,100,200])
        ax_bis.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)

        plt.savefig(folder+"/Comparison_RCCvsGC."+fmt, dpi=dpi, format=fmt)

    ### Comparison of RCC in positive and negative lags
    if (4 or 0) in figs_to_do:
        fig, ax = plt.subplots(figsize=(10,20))
        ax.remove()
        
        ## SIM 1
        plt.gcf().text(0.41, 0.98, "Sim 1", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.825 , 0.425, 0.15]
        ax1 = fig.add_axes([left, bottom, width, height])
        ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)    
        yticks, space = [], 0.15
        for pp in range(1,4):
            label_neg = r"$\tau<0$" if pp==3 else None
            label_pos = r"$\tau>0$" if pp==3 else None
            ax1.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[0][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1], label=label_neg)
            ax1.plot(np.where(lags<0, lags, np.nan), np.flip(np.where(lags>0, results_RCC[0][-1][:,pp]-space*(1-pp), np.nan)), linestyle='dashed', linewidth=2, color=colors[pp-1], label=label_pos)
            ax1.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax1.set_yticks(yticks), ax1.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax1.set_xticks(range(-30,1,5)), ax1.set_xticklabels(np.abs(range(-30,1,5)))
        ax1.legend(frameon=True, fontsize=15, ncols=2, bbox_to_anchor=(0.55, 0.675, 0.2, 0.5))
        ax1.get_legend().legend_handles[0].set_color('black'), ax1.get_legend().legend_handles[1].set_color('black')
        ax1.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.825 , 0.425, 0.15]
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.spines["top"].set_visible(False), ax2.spines["right"].set_visible(False)    
        labels, max_lag, space_x = ["PPV", "NPV", "AUC"], 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[lags<0], results_RCC[0][-1][:,pp][lags<0], y=np.flip(results_RCC[0][-1][:,pp][lags>0]), max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[0][-1][:,pp+3][lags<0], delta_y=np.flip(results_RCC[0][-1][:,pp+3][lags>0])
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax2.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], label=labels[pp-1], markersize=10, alpha=0.5)
            ax2.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                ax2.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax2.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax2.set_xticks(cut_lag), ax2.set_xticklabels(np.abs(cut_lag)), ax2.set_ylim([-10,100])
        ax2.legend(frameon=True, fontsize=15, ncols=3, bbox_to_anchor=(0.85, 0.675, 0.2, 0.5))
        ax2.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15), ax2.set_ylabel("Relative Increase (%)", fontsize=15)

        ## SIM 7
        plt.gcf().text(0.45, 0.78, "Sim 7", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.625 , 0.425, 0.15]
        ax3 = fig.add_axes([left, bottom, width, height])
        ax3.spines["top"].set_visible(False), ax3.spines["right"].set_visible(False)
        yticks, space = [], 0.5
        for pp in range(1,4):
            ax3.plot(np.where(results_RCC[1][-1][:,0]<0, results_RCC[1][-1][:,0], np.nan), np.where(results_RCC[1][-1][:,0]<0, results_RCC[1][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax3.plot(np.where(results_RCC[1][-1][1:,0]<0, results_RCC[1][-1][1:,0], np.nan), np.flip(np.where(results_RCC[1][-1][1:,0]>0, results_RCC[1][-1][1:,pp]-space*(1-pp), np.nan)), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax3.hlines(y=space*(pp-1)+0.5, xmin=results_RCC[1][-1][0,0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax3.set_yticks(yticks), ax3.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=75, va="center", fontsize=15)
        ax3.set_xticks(np.where(results_RCC[1][-1][:,0]<0, results_RCC[1][-1][:,0], np.nan))
        ax3.set_xticklabels(np.abs(np.where(results_RCC[1][-1][:,0]<0, results_RCC[1][-1][:,0], np.nan)))
        ax3.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.625 , 0.425, 0.15]
        ax4 = fig.add_axes([left, bottom, width, height])
        ax4.spines["top"].set_visible(False), ax4.spines["right"].set_visible(False)    
        max_lag, space_x = 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                results_RCC[1][-1][1:,0][results_RCC[1][-1][1:,0]<0], results_RCC[1][-1][1:,pp][results_RCC[1][-1][1:,0]<0], y=np.flip(results_RCC[1][-1][:,pp][results_RCC[1][-1][:,0]>0]),
                max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[1][-1][1:,pp+3][results_RCC[1][-1][1:,0]<0], delta_y=np.flip(results_RCC[1][-1][:,pp+3][results_RCC[1][-1][:,0]>0])
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax4.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], markersize=10, alpha=0.5)
            ax4.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                ax4.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax4.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax4.set_xticks(cut_lag), ax4.set_xticklabels(np.abs(cut_lag)), ax4.set_ylim([-10,90])
        ax4.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15), ax4.set_ylabel("Relative Increase (%)", fontsize=15)

        ## SIM 15
        plt.gcf().text(0.45, 0.58, "Sim 15", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.425 , 0.425, 0.15]
        ax5 = fig.add_axes([left, bottom, width, height])
        ax5.spines["top"].set_visible(False), ax5.spines["right"].set_visible(False)
        yticks, space = [], 0.2
        for pp in range(1,4):
            ax5.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[2][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax5.plot(np.where(lags<0, lags, np.nan), np.flip(np.where(lags>0, results_RCC[2][-1][:,pp]-space*(1-pp), np.nan)), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax5.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax5.set_yticks(yticks), ax5.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax5.set_xticks(range(-30,1,5)), ax5.set_xticklabels(np.abs(range(-30,1,5)))
        ax5.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.425 , 0.425, 0.15]
        ax6 = fig.add_axes([left, bottom, width, height])
        ax6.spines["top"].set_visible(False), ax6.spines["right"].set_visible(False)    
        max_lag, space_x = 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[lags<0], results_RCC[2][-1][:,pp][lags<0], y=np.flip(results_RCC[2][-1][:,pp][lags>0]), max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[2][-1][:,pp+3][lags<0], delta_y=np.flip(results_RCC[2][-1][:,pp+3][lags>0])
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax6.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], markersize=10, alpha=0.5)
            ax6.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                ax6.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax6.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax6.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15), ax6.set_ylabel("Relative Increase (%)", fontsize=15)

        ## SIM 19
        plt.gcf().text(0.45, 0.38, "Sim 19", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.225 , 0.425, 0.15]
        ax7 = fig.add_axes([left, bottom, width, height])
        ax7.spines["top"].set_visible(False), ax7.spines["right"].set_visible(False)
        yticks, space = [], 0.7
        for pp in range(1,4):
            ax7.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[3][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax7.plot(np.where(lags<0, lags, np.nan), np.flip(np.where(lags>0, results_RCC[3][-1][:,pp]-space*(1-pp), np.nan)), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax7.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax7.set_yticks(yticks), ax7.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax7.set_xticks(range(-30,1,5)), ax7.set_xticklabels(np.abs(range(-30,1,5)))
        ax7.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.225 , 0.425, 0.15]
        ax8 = fig.add_axes([left, bottom, width, height])
        ax8.spines["top"].set_visible(False), ax8.spines["right"].set_visible(False)    
        max_lag, space_x = 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[lags<0], results_RCC[3][-1][:,pp][lags<0], y=results_RCC[3][-1][:,pp][lags>0], max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[3][-1][:,pp+3][lags<0], delta_y=results_RCC[3][-1][:,pp+3][lags>0]
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax8.plot(h_data, np.where(rel_increase<200,rel_increase,np.nan), 'o-', linewidth=0.8, color=colors[pp-1], markersize=10, alpha=0.5)
            ax8.errorbar(h_data, rel_increase, yerr=np.where(error<100,error,np.nan), ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                if sm != (len(significance)-1):
                    ax8.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax8.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax8.set_xticks(cut_lag), ax8.set_xticklabels(np.abs(cut_lag)), ax8.set_ylim([-40,40])
        ax8.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15), ax8.set_ylabel("Relative Increase (%)", fontsize=15)

        ## SIM 28
        plt.gcf().text(0.45, 0.19, "Sim 28", fontsize=30, fontweight="bold")
        left, bottom, width, height = [0.06, 0.025 , 0.425, 0.15]
        ax9 = fig.add_axes([left, bottom, width, height])
        ax9.spines["top"].set_visible(False), ax9.spines["right"].set_visible(False)
        yticks, space = [], 0.2
        for pp in range(1,4):
            ax9.plot(np.where(lags<0, lags, np.nan), np.where(lags<0, results_RCC[4][-1][:,pp]-space*(1-pp), np.nan), linestyle='solid', linewidth=2, color=colors[pp-1])
            ax9.plot(np.where(lags<0, lags, np.nan), np.flip(np.where(lags>0, results_RCC[4][-1][:,pp]-space*(1-pp), np.nan)), linestyle='dashed', linewidth=2, color=colors[pp-1])
            ax9.hlines(y=space*(pp-1)+0.5, xmin=lags[0], xmax=0, color="black", linestyle='dotted', linewidth=1)
            yticks.append(space*(pp-1)+0.5)
        ax9.set_yticks(yticks), ax9.set_yticklabels(["PPV=0.5","NPV=0.5","AUC=0.5"], rotation=70, va="center", fontsize=15)
        ax9.set_xticks(range(-30,1,5)), ax9.set_xticklabels(np.abs(range(-30,1,5)))
        ax9.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15)

        left, bottom, width, height = [0.55, 0.025 , 0.4, 0.15]
        ax9 = fig.add_axes([left, bottom, width, height])
        ax9.spines["top"].set_visible(False), ax9.spines["right"].set_visible(False)  
        labels, max_lag, space_x = ["PPV", "NPV", "AUC"], 5, 0.2
        for pp in range(1,4):
            cut_lag, rel_increase, error, significance = relative_increase(
                lags[lags<0], results_RCC[4][-1][:,pp][lags<0], y=np.flip(results_RCC[4][-1][:,pp][lags>0]), max_lag=max_lag, include_positive=False,
                delta_x=results_RCC[4][-1][:,pp+3][lags<0], delta_y=np.flip(results_RCC[4][-1][:,pp+3][lags>0])
            )
            h_data = cut_lag + (2 - pp) * space_x
            ax9.plot(h_data, rel_increase, 'o-', linewidth=0.8, color=colors[pp-1], label=labels[pp-1], markersize=10, alpha=0.5)
            ax9.errorbar(h_data, rel_increase, yerr=error, ecolor=colors[pp-1], capsize=5, fmt='none')
            for sm, s_mark in enumerate(significance):
                ax9.text(h_data[sm]-0.065, rel_increase[sm] + error[sm] + 3.5, s_mark, fontsize=15, fontweight="bold", rotation='vertical')
        ax9.hlines(y=0, xmin=cut_lag[0]-space_x, xmax=-(1-space_x), color="black", linestyle='dotted', linewidth=1)
        ax9.set_xticks(cut_lag), ax9.set_xticklabels(np.abs(cut_lag)), ax9.set_ylim([-10,150])
        ax9.set_xlabel(r"$|\tau|$"+"(step)", fontsize=15), ax9.set_ylabel("Relative Increase (%)", fontsize=15)

        plt.savefig(folder+"/RCC_positive-vs-negative."+fmt, dpi=dpi, format=fmt)