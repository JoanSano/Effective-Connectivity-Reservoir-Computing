import matplotlib.pylab as plt
import numpy as np
import os
import pandas as pd 

def plot_evidence(lags, *to_plot, **kwargs):
    """
    TODO: Add description

    Arguments
    ---------
    lags:
    *to_plot: (dicts) where the keys ["data", "error", "label", "color", "style", "linewidth"]
    """

    # Location of the maximum correlation
    if 'scale' in kwargs.keys():
        lags = lags * kwargs['scale'] # Repetition Time (TR)

    # Instantiate figure
    fig, ax = plt.subplots(figsize=(6,4))
    ax.remove()

    ##################
    left, bottom, width, height = [0.1, 0.12 , 0.85, 0.85]
    ax1 = fig.add_axes([left, bottom, width, height])
    # Plot main curves
    for curve in to_plot:    
        ax1.plot(lags, curve["data"], curve["linewidth"], color=curve["color"], linestyle=curve["style"], label=curve["label"], alpha=curve["alpha"])
        ax1.fill_between(lags, curve["data"]-curve["error"], curve["data"]+curve["error"],
            alpha=0.2, edgecolor=curve["color"], facecolor=curve["color"], linewidth=0        
        )
    # Plot significant times: It requires 
    if "significance_marks" in kwargs.keys():
        y_ini = -0.01
        for curve in kwargs["significance_marks"]:
            ax1.fill_between(curve["data"]*lags, y_ini, y_ini+0.02, alpha=0.8, facecolor=curve["color"], linewidth=0, label=curve["label"])
            y_ini += 0.02
    # Figures details
    z_min, z_max = kwargs["limits"]
    ax1.vlines(x=0, ymin=z_min, ymax=z_max, linewidth=0.3, color='grey', linestyles='--')
    ax1.hlines(y=0, xmin=lags[0], xmax=lags[-1], linewidth=0.3, color='grey', linestyles='--')
    ax1.spines["top"].set_visible(False), ax1.spines["right"].set_visible(False)
    ax1.set_ylabel(kwargs['y_label'], fontsize=15)
    ax1.set_yticks([z_min, 0.5*(z_min+z_max),z_max]), ax1.set_yticklabels([str(z_min), str(0.5*(z_min+z_max)), str(z_max)]), ax1.set_ylim([z_min-0.01,z_max+0.02])
    ax1.set_xlim([lags[0],lags[-1]]), ax1.set_xlabel(kwargs['x_label'], fontsize=15)
    # Legend
    plt.legend(fontsize=8, frameon=False, ncols=2)
    
    ###############
    # Visualization
    if 'save' in kwargs.keys():
        format = kwargs['save'].split(".")[-1]
        if 'dpi' in kwargs.keys():
            plt.savefig(kwargs['save'], dpi=kwargs['dpi'], format=format)
        else:
            plt.savefig(kwargs['save'], format=format)
    else:
        plt.show()
    plt.close()

def generate_report(
        output_dir, name_subject, roi_i, roi_j,
        lags, i2j, j2i, surrogate_i2j, surrogate_j2i,
        Score_i2j, Score_j2i, Score_ij, 
        evidence_i2j, evidence_j2i, evidence_ij,
        plot=True, format='svg'
    ):   
    """
    TODO: Add description

    Arguments
    ---------
    lags:
    """         
    # Means and Standard Errors of the Mean
    mean_i2j, sem_i2j = np.mean(i2j, axis=1), np.std(i2j, axis=1) / np.sqrt(i2j.shape[1])
    mean_j2i, sem_j2i = np.mean(j2i, axis=1), np.std(j2i, axis=1) / np.sqrt(j2i.shape[1])
    mean_i2js, sem_i2js = np.mean(surrogate_i2j, axis=1), np.std(surrogate_i2j, axis=1) / np.sqrt(surrogate_i2j.shape[1])
    mean_j2is, sem_j2is = np.mean(surrogate_j2i, axis=1), np.std(surrogate_j2i, axis=1) / np.sqrt(surrogate_j2i.shape[1])

    # Destination directories and names of outputs
    output_dir_subject = os.path.join(output_dir,name_subject)
    numerical = os.path.join(output_dir_subject,"Numerical")
    figures = os.path.join(output_dir_subject,"Figures")
    if not os.path.exists(output_dir_subject):
        os.mkdir(output_dir_subject)
    if not os.path.exists(numerical):
        os.mkdir(numerical)
    if not os.path.exists(figures):
        os.mkdir(figures)
    name_subject_RCC = name_subject + '_RCC_rois-' +str(roi_i+1) + 'vs' + str(roi_j+1)
    name_subject_RCC_figure = os.path.join(figures, name_subject_RCC+'.' + format)
    name_subject_RCC_numerical = os.path.join(numerical ,name_subject_RCC+'.tsv')

    # Save numerical results
    i2jlabel, j2ilabel = str(roi_i+1) + ' --> ' + str(roi_j+1), str(roi_j+1) + ' --> ' + str(roi_i+1)
    ijlabel = str(roi_i+1) + ' <--> ' + str(roi_j+1)
    results = pd.DataFrame({
        "time-lags": lags,
        "Score " + ijlabel: Score_ij,
        "Score " + i2jlabel: Score_i2j,
        "Score " + j2ilabel: Score_j2i,
        "Evidence " + ijlabel: evidence_ij,
        "Evidence " + i2jlabel: evidence_i2j,
        "Evidence " + j2ilabel: evidence_j2i,
        i2jlabel: mean_i2j,
        j2ilabel: mean_j2i,
        'SEM ' + i2jlabel: sem_i2j,
        'SEM ' + j2ilabel: sem_j2i,
        i2jlabel + 'Surrogate': mean_i2js,
        j2ilabel + 'Surrogate': mean_j2is,
        i2jlabel + 'Surrogate SEM ': sem_i2js,
        j2ilabel + 'Surrogate SEM ': sem_j2is
    })
    results.to_csv(name_subject_RCC_numerical, index=False, sep='\t', decimal='.')

    if plot:
        plot_evidence(
            lags,
            {"data": mean_i2j, "error": sem_i2j, "label": r"$\rho_{\tau}$"+f"({str(roi_i+1)},{str(roi_j+1)})", "color": "darkorange", "style": "-", "linewidth": 1, "alpha": 1}, 
            {"data": mean_j2i, "error": sem_j2i, "label": r"$\rho_{\tau}$"+f"({str(roi_j+1)},{str(roi_i+1)})", "color": "green", "style": "-", "linewidth": 1, "alpha": 1}, 
            {"data": mean_i2js, "error": sem_i2js, "label": r"$\rho_{\tau}$"+f"({str(roi_i+1)},{str(roi_j+1)}"+r"$_{S}$"+")", "color": "bisque", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
            {"data": mean_j2is, "error": sem_j2is, "label": r"$\rho_{\tau}$"+f"({str(roi_j+1)},{str(roi_i+1)}"+r"$_{S}$"+")", "color": "lightgreen", "style": "-", "linewidth": 0.7, "alpha": 0.5}, 
            save=name_subject_RCC_figure, dpi=300, y_label="Scores", x_label=r"$\tau$"+"(steps)", limits=(0,1), #scale=0.720, 
            significance_marks=[
                {"data": evidence_i2j, "color": "blue", "label": i2jlabel},
                {"data": evidence_j2i, "color": "red", "label": j2ilabel},
                {"data": evidence_ij, "color": "purple", "label": ijlabel}
            ]
        ) 

if __name__ == '__main__':
    pass