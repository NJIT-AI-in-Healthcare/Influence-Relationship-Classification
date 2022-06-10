#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 23:07:37 2019

@author: LMD
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import random
from spyder_kernels.utils.iofuncs import load_dictionary


data = load_dictionary("../partial_results/roc_all_final.spydata")[0]
fig = plt.figure(1)
plt.plot(data['fpr_baseline'], data['tpr_baseline'], label="Baseline")
plt.plot(
    data['fpr_cat_match_pyramid'],
    data['tpr_cat_match_pyramid'],
    label="MatchPyramid+cat"
    )
plt.plot(
    data['fpr_dot_match_pyramid'],
    data['tpr_dot_match_pyramid'],
    label="MatchPyramid+dot"
    )
plt.plot(data['fpr_cat_arci'], data['tpr_cat_arci'], label="ARC-I+cat")
plt.plot(data['fpr_dot_arci'], data['tpr_dot_arci'], label="ARC-I+dot")
plt.xlabel('False positive rate', fontsize=20)
plt.ylabel('True positive rate', fontsize=20)
plt.legend(loc=0, fontsize=15)
fig.set_size_inches(12, 12)
fig.savefig('../partial_results/ROC.png')
fig.show()
