# %%
import mat73
import h5py
import hdf5storage as st
# from pymatreader import read_mat
import pickle
import os
from time import time
import gzip

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap

from scipy.stats import wilcoxon, norm, kruskal, tukey_hsd, mannwhitneyu, sem
from scipy.optimize import curve_fit
from scipy.io import loadmat
from scipy.linalg import null_space
from scipy.spatial.distance import cdist

import seaborn as sns
from copy import deepcopy as dc
from statsmodels.stats.multitest import multipletests
import math
from itertools import combinations, product

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# from libsvm import svmutil

# import torch
# import torch.nn as nn
# import torch.optim as optim

# from pycaret.classification import *

# %%
def compute_mean_var_trial(label_cnt_dict, rate_sorted):    
    list_trial_mean = [[0]] * len(label_cnt_dict)
    list_trial_var = [[0]] * len(label_cnt_dict)

    for trial_ind, trial_type in enumerate(label_cnt_dict):
        
        trial_rate = np.array(rate_sorted.loc[:, trial_type])                
        trial_mean = np.mean(trial_rate, axis=1)
        trial_var = np.var(trial_rate, axis=1, ddof=1)

        trial_mean = pd.DataFrame(trial_mean, columns=[trial_type], index=rate_sorted.index)
        trial_var = pd.DataFrame(trial_var, columns=[trial_type], index=rate_sorted.index)
        list_trial_mean[trial_ind] = pd.concat([trial_mean] * label_cnt_dict[trial_type], axis=1)
        list_trial_var[trial_ind] = pd.concat([trial_var] * label_cnt_dict[trial_type], axis=1)

    rate_sorted_mean = pd.concat(list_trial_mean, axis=1)
    rate_sorted_var = pd.concat(list_trial_var, axis=1)

    return rate_sorted_mean, rate_sorted_var

# %%
def compute_mean_var_trial_collapse(label_cnt_dict, rate_sorted):    
    list_trial_mean = [[0]] * len(label_cnt_dict)
    list_trial_var = [[0]] * len(label_cnt_dict)

    for trial_ind, trial_type in enumerate(label_cnt_dict):
        
        trial_rate = np.array(rate_sorted.loc[:, trial_type])                
        trial_mean = np.mean(trial_rate, axis=1, dtype=np.longdouble)
        trial_var = np.var(trial_rate, axis=1, ddof=1, dtype=np.longdouble)

        trial_mean = pd.DataFrame(trial_mean, columns=[trial_type], index=rate_sorted.index)
        trial_var = pd.DataFrame(trial_var, columns=[trial_type], index=rate_sorted.index)
        list_trial_mean[trial_ind] = trial_mean.copy()
        list_trial_var[trial_ind] = trial_var.copy()

    rate_sorted_mean = pd.concat(list_trial_mean, axis=1)
    rate_sorted_var = pd.concat(list_trial_var, axis=1)

    return rate_sorted_mean, rate_sorted_var

# %%
# overlap consistency 
def compute_overlap_stimpairs_consis(sess_ind):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = [0, 1, 2]
    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    num_sampling = 10 # Number of random neuron partitioning

    print(f'sess_ind: {sess_ind}')
    
    rate_sorted = list_rate_all[sess_ind].sort_index(axis=1)
    stm = rate_sorted.columns.copy()
    num_neurons = rate_sorted.shape[0]
    
    np.random.seed(0)

    # Multiply by delta t to convert to spike counts
    rate_sorted = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()

    # exclude neurons with low firing rate in every stimulus
    thres_fr = 4 # Hz
    bool_neu = np.any(rate_sorted_mean_coll >= thres_fr * 0.25, axis=1)
    bool_neu = np.random.choice(rate_sorted.index, np.sum(bool_neu), replace=False) # random subsample to match dimensionality
    rate_sorted = rate_sorted.loc[bool_neu].reset_index(drop=True)

    list_overlap_asis2 = np.full((num_sampling, 2, num_trial_types, num_trial_types), np.nan) # 2 neuronal subsets
    list_overlap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan)
    list_gap_asis2 = np.full((num_sampling, 2, num_trial_types, num_trial_types), np.nan) # 2 neuronal subsets
    list_gap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan)
    for sampling_ind in range(num_sampling):
        # print(f'sess_ind = {sess_ind}, sampling ind = {sampling_ind}')

        # Partition neurons
        neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
        neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 partitioning
        neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
        if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
            neu_div_inds2 = neu_div_inds2[:-1].copy()        
        list_neu_div_inds = dc([neu_div_inds1, neu_div_inds2])
        
        # Calculate overlap for all stimulus pairs

        # Determine criteria using internal pairwise distance for each stimulus
        list_pwdist = np.zeros((2, num_trial_types, 2))
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            pwdist_tt1 = cdist(rate_sorted.loc[neu_div_inds1, trial_type].T, rate_sorted.loc[neu_div_inds1, trial_type].T, 'euclidean')
            pwdist_tt2 = cdist(rate_sorted.loc[neu_div_inds2, trial_type].T, rate_sorted.loc[neu_div_inds2, trial_type].T, 'euclidean')
            pwdist_tt1[np.diag_indices(rate_sorted.loc[neu_div_inds1, trial_type].shape[1])] = np.nan
            pwdist_tt2[np.diag_indices(rate_sorted.loc[neu_div_inds2, trial_type].shape[1])] = np.nan
            list_pwdist[0, trial_type_ind, 0] = np.nanpercentile(pwdist_tt1, 5)
            list_pwdist[0, trial_type_ind, 1] = np.nanmean(pwdist_tt1)
            list_pwdist[1, trial_type_ind, 0] = np.nanpercentile(pwdist_tt2, 5)
            list_pwdist[1, trial_type_ind, 1] = np.nanmean(pwdist_tt2)

        list_overlap_asis = np.full((2, num_trial_types, num_trial_types), np.nan) # 2 neuronal subsets
        list_gap_asis = np.full((2, num_trial_types, num_trial_types), np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            for div_ind in range(2):
                n_neighbors = 5
                nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                
                rate_tt = rate_sorted.loc[list_neu_div_inds[div_ind], trial_type].copy()
                rate_rest = rate_sorted.loc[list_neu_div_inds[div_ind], all_stm_unique[all_stm_unique != trial_type]].copy()
                rate_rest = rate_sorted.loc[list_neu_div_inds[div_ind]].copy()
                nbrs.fit(rate_tt.T)
                nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
                nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

                pwdist_temp = list_pwdist[div_ind, trial_type_ind, 0].copy()
                overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
                # list_overlap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = overlap.copy()
                list_overlap_asis[div_ind, trial_type_ind] = overlap.copy()

                # gap = np.min(nbr_dist, axis=1).groupby(level=0).min()
                # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                # # print(gap.shape)
                # # list_gap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[div_ind, all_stm_unique != trial_type, 1]
                # list_gap_asis[div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]
                        
        list_overlap_asis2[sampling_ind] = list_overlap_asis.copy()
        list_gap_asis2[sampling_ind] = list_gap_asis.copy()

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
        
        list_overlap_RRneuron2 = np.full((len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan) # 2 neuronal subsets
        list_gap_RRneuron2 = np.full((len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan) # 2 neuronal subsets
        for slope_ind, target_slope in enumerate(list_target_slopes):
            start_time = time()
            # print(f'sampling ind = {sampling_ind}, target slope = {target_slope:.1f}')

            # calculate target variance
            var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                    columns=rate_sorted_var_coll.columns)
            for trial_type in rate_sorted_var_coll.columns:
                var_estim_dr.loc[:, trial_type] = \
                    np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
            # var_estim_dr = np.repeat(var_estim_dr, all_label_counts, axis=1)
            # print(var_estim_dr)

            # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed
            offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.log10(rate_sorted_mean_coll).mean(axis=0) + list_slopes_dr.iloc[1, :]) 

            var_rs_noisy = \
                pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
                    .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # collapsed
            var_rs_noisy = np.repeat(var_rs_noisy.values, all_stm_counts, axis=1)

            # Compute changed residual and add back to the mean            
            rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                .mul(np.sqrt(var_rs_noisy))
            # print(rate_resid_RRneuron_dr)
            rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # convert NaN to 0!

            # Calculate overlap for all stimulus pairs

            # Determine criteria using internal pairwise distance for each stimulus
            list_pwdist = np.zeros((2, num_trial_types, 2))
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                pwdist_tt1 = cdist(rate_RRneuron_dr.loc[neu_div_inds1, trial_type].T, rate_RRneuron_dr.loc[neu_div_inds1, trial_type].T, 'euclidean')
                pwdist_tt2 = cdist(rate_RRneuron_dr.loc[neu_div_inds2, trial_type].T, rate_RRneuron_dr.loc[neu_div_inds2, trial_type].T, 'euclidean')
                pwdist_tt1[np.diag_indices(rate_RRneuron_dr.loc[neu_div_inds1, trial_type].shape[1])] = np.nan
                pwdist_tt2[np.diag_indices(rate_RRneuron_dr.loc[neu_div_inds2, trial_type].shape[1])] = np.nan
                list_pwdist[0, trial_type_ind, 0] = np.nanpercentile(pwdist_tt1, 5)
                list_pwdist[0, trial_type_ind, 1] = np.nanmean(pwdist_tt1)
                list_pwdist[1, trial_type_ind, 0] = np.nanpercentile(pwdist_tt2, 5)
                list_pwdist[1, trial_type_ind, 1] = np.nanmean(pwdist_tt2)

            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                for div_ind in range(2):
                    n_neighbors = 5
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                    
                    rate_tt = rate_RRneuron_dr.loc[list_neu_div_inds[div_ind], trial_type].copy()
                    rate_rest = rate_RRneuron_dr.loc[list_neu_div_inds[div_ind], all_stm_unique[all_stm_unique != trial_type]].copy()
                    rate_rest = rate_RRneuron_dr.loc[list_neu_div_inds[div_ind]].copy()
                    nbrs.fit(rate_tt.T)
                    nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
                    nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

                    pwdist_temp = list_pwdist[div_ind, trial_type_ind, 0].copy()
                    overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
                    # list_overlap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = overlap.copy()
                    list_overlap_RRneuron2[slope_ind, div_ind, trial_type_ind] = overlap.copy()

                    # gap = np.min(nbr_dist, axis=1).groupby(level=0).min()
                    # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                    # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                    # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                    # # print(gap.shape)
                    # # list_gap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[div_ind, all_stm_unique != trial_type, 1]
                    # list_gap_RRneuron2[slope_ind, div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]
            print(f'sess_ind {sess_ind}, sampling_ind {sampling_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

        list_overlap_RRneuron3[sampling_ind] = list_overlap_RRneuron2.copy()
        list_gap_RRneuron3[sampling_ind] = list_gap_RRneuron2.copy()

    # Save into a file
    filename = 'overlap_nbr_stimpairs_consis_ABO_' + str(sess_ind) +  '.pickle.gz'
    filename = 'overlap_nbr_stimpairs_consis_ABO_fr_' + str(sess_ind) +  '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_overlap_asis2', 'list_overlap_RRneuron3', 'list_gap_asis2', 'list_gap_RRneuron3'],
                     'list_overlap_asis2': list_overlap_asis2, 'list_overlap_RRneuron3': list_overlap_RRneuron3,
                     'list_gap_asis2': list_gap_asis2, 'list_gap_RRneuron3': list_gap_RRneuron3}, f)
                
    print("Ended Process", c_proc.name)

# %%
# overlap of stimulus pairs 
def compute_overlap_stimpairs(sess_ind, box_size=None):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    rng = np.random.default_rng(sess_ind)

    # print(f'sess_ind: {sess_ind}')
    
    rate_sorted = list_rate_all[sess_ind].sort_index(axis=1)
    stm = rate_sorted.columns.copy()
    num_neurons = rate_sorted.shape[0]

    # Multiply by delta t to convert to spike counts
    rate_sorted = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()

    # # exclude neurons with low firing rate in every stimulus
    # thres_fr = 4 # Hz
    # bool_neu = np.any(rate_sorted_mean_coll >= thres_fr * 0.25, axis=1)
    # bool_neu = rng.choice(rate_sorted.index, np.sum(bool_neu), replace=False) # random subsample to match dimensionality
    # rate_sorted = rate_sorted.loc[bool_neu]

    # trial shuffling
    rate_shuf = pd.DataFrame(np.zeros_like(rate_sorted), columns=rate_sorted.columns)
    for neu_ind in range(rate_sorted.shape[0]):
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            shuf_inds = rng.permutation(rate_sorted.loc[:, trial_type].shape[1])
            rate_shuf.loc[neu_ind, trial_type] = rate_sorted.loc[neu_ind, trial_type].iloc[shuf_inds]
    rate_sorted = rate_shuf.copy()

    # # inter-centroid geodesic distance

    # # concatenate centroids of all stimuli
    # rate_plus_mean = pd.concat([rate_sorted, rate_sorted_mean_coll], axis=1)

    # # Compute geodesic distance matrix
    # n_components = 1 # target number of dimensions
    # # n_components = rate.shape[0] # target number of dimensions
    # n_neighbors = 5 # number of neighbors

    # isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    
    # isomap.fit(rate_plus_mean.T)
    # mean_dist_mat_asis = isomap.dist_matrix_[rate_sorted.shape[1]:, rate_sorted.shape[1]:].copy() # inter-centroid geodesic distance matrix

    # Determine criteria using internal pairwise distance for each stimulus
    pwdist_thr = 5
    list_pwdist = np.zeros((num_trial_types, 3))
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        pwdist = cdist(rate_sorted.loc[:, trial_type].T, rate_sorted.loc[:, trial_type].T, 'euclidean')
        pwdist[np.diag_indices(rate_sorted.loc[:, trial_type].shape[1])] = np.nan
        list_pwdist[trial_type_ind, 0] = np.nanpercentile(pwdist, pwdist_thr)
        list_pwdist[trial_type_ind, 1] = np.nanpercentile(pwdist, 100-pwdist_thr)
        list_pwdist[trial_type_ind, 2] = np.nanmean(pwdist)

    dist_thr = 10
    list_overlap_asis = np.full((num_trial_types, num_trial_types), np.nan)
    list_gap_asis = np.full((num_trial_types, num_trial_types), np.nan)
    list_bool_close = np.empty((num_trial_types, num_trial_types), dtype=object)
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        n_neighbors = 5
        # n_neighbors = all_stm_counts[trial_type_ind]
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        
        rate_tt = rate_sorted.loc[:, trial_type].copy()
        # rate_rest = rate_sorted.loc[:, all_stm_unique[all_stm_unique != trial_type]].copy()
        rate_rest = rate_sorted.copy()
        nbrs.fit(rate_tt.T)
        nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
        nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)
        # print(nbr_dist)

        # 3. % (knn <= threshold)
        # overlap = (np.mean(nbr_dist, axis=1) <= dist_thr).groupby(level=0).mean()
        pwdist_temp = list_pwdist[trial_type_ind, 0].copy()
        overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
        # # print(np.isnan(overlap).sum())
        # list_overlap_asis[trial_type_ind, all_stm_unique != trial_type] = overlap.copy()
        list_overlap_asis[trial_type_ind] = overlap.copy()

        # gap = np.min(nbr_dist, axis=1).groupby(level=0).quantile(0.05)
        # # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
        # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
        # # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().quantile(0.05, axis=1)
        # # print(gap.shape)
        # # list_gap_asis[trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[all_stm_unique != trial_type, 1]
        # list_gap_asis[trial_type_ind] = gap/list_pwdist[:, 2]

    # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
    rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
    rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
    
    list_overlap_RRneuron2 = np.zeros((len(list_target_slopes), len(list(combinations(range(num_trial_types), 2))), num_neurons, 2))
    list_overlap_RRneuron2 = np.full((len(list_target_slopes), num_trial_types, num_trial_types), np.nan)
    list_gap_RRneuron2 = np.full((len(list_target_slopes), num_trial_types, num_trial_types), np.nan)
    # list_overlap_RRneuron2 = np.full((len(list_target_slopes), len(list(combinations(range(num_trial_types), 2)))), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        start_time = time()
        # print(f'sess_ind {sess_ind}, target slope {target_slope:.1f}')

        # calculate target variance
        var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                columns=rate_sorted_var_coll.columns)
        for trial_type in rate_sorted_var_coll.columns:
            var_estim_dr.loc[:, trial_type] = \
                np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
        # var_estim_dr = np.repeat(var_estim_dr, all_label_counts, axis=1)
        # print(var_estim_dr)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed
        offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.log10(rate_sorted_mean_coll).mean(axis=0) + list_slopes_dr.iloc[1, :])

        var_rs_noisy = \
            pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
                .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # collapsed
        var_rs_noisy = np.repeat(var_rs_noisy.values, all_stm_counts, axis=1)

        # Compute changed residual and add back to the mean            
        rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            .mul(np.sqrt(var_rs_noisy))
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # convert NaN to 0!

        # Determine criteria using internal pairwise distance for each stimulus
        list_pwdist = np.zeros((num_trial_types, 3))
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            pwdist = cdist(rate_RRneuron_dr.loc[:, trial_type].T, rate_RRneuron_dr.loc[:, trial_type].T, 'euclidean')
            pwdist[np.diag_indices(rate_RRneuron_dr.loc[:, trial_type].shape[1])] = np.nan
            list_pwdist[trial_type_ind, 0] = np.nanpercentile(pwdist, pwdist_thr)
            list_pwdist[trial_type_ind, 1] = np.nanpercentile(pwdist, 100-pwdist_thr)
            list_pwdist[trial_type_ind, 2] = np.nanmean(pwdist)

        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            n_neighbors = 5
            # n_neighbors = all_stm_counts[trial_type_ind]
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            
            rate_tt = rate_RRneuron_dr.loc[:, trial_type].copy()
            # rate_rest = rate_RRneuron_dr.loc[:, all_stm_unique[all_stm_unique != trial_type]].copy()
            rate_rest = rate_RRneuron_dr.copy()
            nbrs.fit(rate_tt.T)
            nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
            nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)
            # print(nbr_dist)

            # 3. % (knn <= threshold)
            # overlap = (np.mean(nbr_dist, axis=1) <= dist_thr).groupby(level=0).mean()
            pwdist_temp = list_pwdist[trial_type_ind, 0].copy()
            overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
            # # print(np.isnan(overlap).sum())
            # list_overlap_RRneuron2[slope_ind, trial_type_ind, all_stm_unique != trial_type] = overlap.copy()
            list_overlap_RRneuron2[slope_ind, trial_type_ind] = overlap.copy()
                       
            # gap = np.min(nbr_dist, axis=1).groupby(level=0).quantile(0.05)
            # # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
            # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
            # # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().quantile(0.05, axis=1)
            # # print(gap.shape)
            # # list_gap_RRneuron2[slope_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[all_stm_unique != trial_type, 1]
            # list_gap_RRneuron2[slope_ind, trial_type_ind] = gap/list_pwdist[:, 2]

        print(f'sess_ind {sess_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'overlap_nbr_stimpairs_ABO_' + str(sess_ind) +  '.pickle.gz'
    filename = 'overlap_nbr_stimpairs_ABO_fr_' + str(sess_ind) +  '.pickle.gz'
    filename = 'overlap_nbr_stimpairs_ABO_shuf_' + str(sess_ind) +  '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_overlap_asis', 'list_overlap_RRneuron2', 'list_gap_asis', 'list_gap_RRneuron2'],
                     'list_overlap_asis': list_overlap_asis, 'list_overlap_RRneuron2': list_overlap_RRneuron2,
                     'list_gap_asis': list_gap_asis, 'list_gap_RRneuron2': list_gap_RRneuron2}, f)
        # pickle.dump({'tree_variables': ['list_overlap_asis', 'list_gap_asis'],
        #              'list_overlap_asis': list_overlap_asis, 'list_gap_asis': list_gap_asis}, f)
                
    print("Ended Process", c_proc.name)

# %%
# loading variables

# ABO Neuropixels
with gzip.open('resp_matrix_ep_RS_all_32sess_allensdk.pickle.gz', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_RS = resp_matrix_ep_RS_all['list_rate_RS'].copy()
    list_rate_RS_dr = resp_matrix_ep_RS_all['list_rate_RS_dr'].copy()
    list_rate_all = resp_matrix_ep_RS_all['list_rate_all'].copy()
    list_rate_all_dr = resp_matrix_ep_RS_all['list_rate_all_dr'].copy()
    list_slopes_RS_an_loglog = resp_matrix_ep_RS_all['list_slopes_RS_an_loglog'].copy()
    list_slopes_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_all_an_loglog'].copy()

    sess_inds_qual_all = resp_matrix_ep_RS_all['sess_inds_qual_all'].copy()

# %%
# multiprocessing

# overlap between stimulus pairs
num_sess = 32
if __name__ == '__main__':

    with mp.Pool() as pool: # set the parameter 'processes' of Pool() if memory error is raised
        list_inputs = [[sess_ind] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_overlap_stimpairs_consis, list_inputs)

# overlap between stimulus pairs
num_sess = 32
if __name__ == '__main__':

    with mp.Pool() as pool: # set the parameter 'processes' of Pool() if memory error is raised
        list_inputs = [[sess_ind] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_overlap_stimpairs, list_inputs)