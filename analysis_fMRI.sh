#!/bin/bash -l

python analysis/analysis_fmri_reconstruction.py Method-RCC_AOMICSchaefer100rsNetDMNPositiveLags-Dataset_Split-70/ Results-AOMIC_rsNet-DMN_Method-RCC --method RCC --length 100 --rsnet DMN
python analysis/analysis_fmri_reconstruction.py Method-RCC_AOMICSchaefer100rsNetDMNNegativeLags-Dataset_Split-70/ Results-AOMIC_rsNet-DMN_Method-RCC --method RCC --length 100 --rsnet DMN
python analysis/analysis_fmri_reconstruction.py Method-GC_AOMICSchaefer100rsNetDMN-Dataset_Split-None/ Results-AOMIC_rsNet-DMN_Method-GC --method GC --length 100 --rsnet DMN