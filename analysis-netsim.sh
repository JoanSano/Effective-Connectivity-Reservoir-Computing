#!/bin/bash -l

#for SIM in 1 7 15 19 28
#do
#python analysis/analysis_netsim_reconstruction.py Method-GC_Netsim-Dataset_Sim-"$SIM"_Split-None Results-Figures_Method-GC_Netsim-Dataset_Sim-"$SIM"_Split-None --method GC --simulation $SIM
#python analysis/analysis_netsim_reconstruction.py Method-RCC_Netsim-Dataset_Sim-"$SIM"_Split-70 Results-Figures_Method-RCC_Netsim-Dataset_Sim-"$SIM"_Split-70 --method RCC --simulation $SIM
#done

python analysis/analysis_netsim_simulations.py --figs 3

#rm Results-RCC_arguments.txt Results-GC_arguments.txt 
#rm Results-Metrics_Method*
#rm -r Results-Figures_Method*Gub12758SanoScience
