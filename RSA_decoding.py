# %%
from pynwb import NWBHDF5IO
from scipy.io import savemat, loadmat
import mat73
import h5py
import hdf5storage as st
# from pymatreader import read_mat
import pickle
import os
import warnings

import multiprocessing as mp
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import wilcoxon, norm, kruskal, tukey_hsd, mode, spearmanr, rankdata
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.special import gammaln

import seaborn as sns
from copy import deepcopy as dc
from statsmodels.stats.multitest import multipletests
from itertools import combinations, product, permutations, combinations_with_replacement
import math
import random
from time import time

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as logit
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, AffinityPropagation, KMeans
from sklearn.decomposition import PCA

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
        trial_mean = np.mean(trial_rate, axis=1, dtype=np.longdouble)
        trial_var = np.var(trial_rate, axis=1, ddof=1, dtype=np.longdouble)

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
# Function to compute cosine similarity
def cos_sim(x, y):
    # x and y are 1D vectors

    # Remove NaN
    x, y = np.array(x), np.array(y)
    bool_notnan = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[bool_notnan].copy(), y[bool_notnan].copy()

    return np.dot(x, y) / (np.linalg.norm(x.astype(np.float32)) * np.linalg.norm(y.astype(np.float32)))

# %%
# Function for column normalization
def normc(matrix):
    '''For 2D matrix'''

    matrix_normalized = matrix / np.linalg.norm(matrix.astype(np.float32), axis=0)

    return matrix_normalized

# %%
# RSA across session pairs (ABO Neuropixels, RRneuron)

def RSA_across_sesspairs_ABO(slope_ind, target_slope, similarity_type='cos_sim'):
    
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # number of trial sampling for each stimulus
    num_trials = 50
    num_trial_types = 119
    num_sess = 32

    print(f'target slope {target_slope:.1f}')

    # Iterate over all sessions

    np.random.seed(0) # match trial order.

    list_RSM_mean_asis = np.zeros((num_sess, num_trial_types, num_trial_types))
    list_RSM_mean_RRneuron = np.zeros((num_sess, num_trial_types, num_trial_types))
    list_rate_RRneuron_dr = np.empty(num_sess, dtype=object)

    # list_sess_inds = np.delete(np.arange(num_sess).astype(int), [0, 6])
    for ind in range(num_sess):

        print(f'ind: {ind}')

        rate = list_rate_all[ind].copy()
        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Multiply by delta t to convert to spike counts
        rate = rate * 0.25

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) 
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # convert to 3D response matrix
        min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
            list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        rate = np.stack(list_rate_tt, axis=2)
        rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
    
        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
        
        list_slopes_dr = list_slopes_all_an_loglog[ind].copy()

        # # trial shuffling
        # rate_shuf = np.zeros_like(rate_sorted)
        # for neu_ind in range(rate_sorted.shape[0]):
        #     shuf_inds = np.random.permutation(rate_sorted.shape[2])
        #     rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() 
        # rate_sorted = rate_shuf.copy()

        # trial order re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        if slope_ind == 0:

            # repeat calculating similarity matrices

            # n_neurons x n_stimuli 2D matrix sampling

            # if ind == 3:
            #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
            tt_pairs = list(combinations(range(min_num_trials), 2))
            n_sampling = np.min([len(tt_pairs), 10000])
            # n_sampling = len(tt_pairs)

            list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
            
            count = 0
            for sampling_ind in range(n_sampling):
                
                rate_sampled_trials1 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][0]]).copy() 
                rate_sampled_trials2 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][1]]).copy()

                # calculate similarity matrix
                RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) 

                # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) 
                list_RSM[sampling_ind] = RSM.copy()
                
                count += 1
                if count % 100 == 0:
                    print(f'count: {count}')

            RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
            list_RSM_mean_asis[ind] = RSM_mean.copy()

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        # Change slope

        # calculate target variance
        var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed 
        offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) 

        var_rs_noisy = \
            pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
        var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

        # Compute changed residual and add back to the mean            
        rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
            * np.sqrt(var_rs_noisy)
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!

        list_rate_RRneuron_dr[ind] = rate_RRneuron_dr.copy()

        # # trial order re-randomization
        # for trial_type_ind in range(num_trial_types):
        #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        # repeat calculating similarity matrices

        # n_neurons x n_stimuli 2D matrix sampling

        # if ind == 3:
        #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
        tt_pairs = list(combinations(range(min_num_trials), 2))
        n_sampling = np.min([len(tt_pairs), 10000])
        # n_sampling = len(tt_pairs)

        list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
        
        count = 0
        for sampling_ind in range(n_sampling):
            
            rate_sampled_trials1 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][0]]).copy() 
            rate_sampled_trials2 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][1]]).copy()

            RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) 

            # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) 
            list_RSM[sampling_ind] = RSM.copy()
            
            count += 1
            if count % 100 == 0:
                print(f'count: {count}')

        RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
        list_RSM_mean_RRneuron[ind] = RSM_mean.copy()

    # Spearman correlation across session pairs
    sess_pairs = list(combinations(range(num_sess), 2))
    list_corr_sesspair_asis = np.zeros((len(sess_pairs), 3)) 
    list_corr_sesspair = np.zeros((len(sess_pairs), 3)) 
    for pair_ind, pair in enumerate(sess_pairs):
        if slope_ind == 0:
            RSM_mean_neu1 = list_RSM_mean_asis[pair[0]].copy()
            RSM_mean_neu2 = list_RSM_mean_asis[pair[1]].copy()

            list_corr_sesspair_asis[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_sesspair_asis[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_sesspair_asis[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
        
        RSM_mean_neu1 = list_RSM_mean_RRneuron[pair[0]].copy()
        RSM_mean_neu2 = list_RSM_mean_RRneuron[pair[1]].copy()

        list_corr_sesspair[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
        bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
        list_corr_sesspair[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
        list_corr_sesspair[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_sesspair_ABO_allneu_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis', 'list_corr_sesspair_asis', 'list_rate_RRneuron_dr', 'list_RSM_mean_RRneuron', 'list_corr_sesspair'], \
                     'list_RSM_mean_asis': list_RSM_mean_asis, 'list_corr_sesspair_asis': list_corr_sesspair_asis,
                     'list_rate_RRneuron_dr': list_rate_RRneuron_dr, 'list_RSM_mean_RRneuron': list_RSM_mean_RRneuron, 'list_corr_sesspair': list_corr_sesspair}, f)

    print("Ended Process", c_proc.name)

# %%
# %%
# RSA across session pairs (ABO Neuropixels, RRneuron)

def RSA_across_sesspairs_ABO_HVA(slope_ind, target_slope, similarity_type='cos_sim'):
    
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # number of trial sampling for each stimulus
    num_trials = 50
    num_trial_types = 119
    num_sess = 32

    np.random.seed(0)

    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']

    print(f'target slope {target_slope:.1f}')

    list_RSM_mean_asis_HVA = {hva: 0 for hva in list_HVA_names}
    list_corr_sesspair_asis_HVA = {hva: 0 for hva in list_HVA_names}

    list_RSM_mean_RRneuron_HVA = {hva: 0 for hva in list_HVA_names}
    list_rate_RRneuron_dr_HVA = {hva: 0 for hva in list_HVA_names}
    list_corr_sesspair_HVA = {hva: 0 for hva in list_HVA_names}
    for area_ind, area in enumerate(list_HVA_names):
        
        # Iterate over all sessions

        list_RSM_mean_asis = np.full((num_sess, num_trial_types, num_trial_types), np.nan)
        list_RSM_mean_RRneuron = np.full((num_sess, num_trial_types, num_trial_types), np.nan)
        list_rate_RRneuron_dr = np.empty(num_sess, dtype=object)

        # list_sess_inds = np.delete(np.arange(num_sess).astype(int), [0, 6])
        for ind in range(num_sess):

            print(f'area: {area}, ind: {ind}')

            rate = list_rate_all_HVA[area][ind].copy()
            if np.any(rate) == True: # if neurons exist

                # rate_sorted = rate.sort_index(axis=1)
                stm = rate.columns.copy()

                # Multiply by delta t to convert to spike counts
                rate = rate * 0.25

                # Create a counting dictionary for each stimulus
                all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) 
                stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
                
                # convert to 3D response matrix
                min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

                list_rate_tt = [None] * num_trial_types
                for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
                    list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

                rate = np.stack(list_rate_tt, axis=2)
                rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
            
                rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
                rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
                    np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
                
                list_slopes_dr = list_slopes_all_an_loglog_HVA[area][ind].copy()

                # trial order re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                if slope_ind == 0:

                    # repeat calculating similarity matrices

                    # n_neurons x n_stimuli 2D matrix sampling

                    # if ind == 3:
                    #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
                    tt_pairs = list(combinations(range(min_num_trials), 2))
                    n_sampling = np.min([len(tt_pairs), 10000])
                    # n_sampling = len(tt_pairs)

                    list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
                    
                    count = 0
                    for sampling_ind in range(n_sampling):
                        
                        rate_sampled_trials1 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][0]]).copy() 
                        rate_sampled_trials2 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][1]]).copy()

                        # calculate similarity matrix
                        RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) 

                        # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) 
                        list_RSM[sampling_ind] = RSM.copy()
                        
                        count += 1
                        if count % 1000 == 0:
                            print(f'count: {count}')

                    RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
                    list_RSM_mean_asis[ind] = RSM_mean.copy()

                # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

                # calculate target variance
                var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

                # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
                # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed 
                offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) 

                var_rs_noisy = \
                    pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                        / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
                var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

                # Compute changed residual and add back to the mean            
                rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
                # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                    * np.sqrt(var_rs_noisy)
                # print(rate_resid_RRneuron_dr)
                rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
                rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!      
                list_rate_RRneuron_dr[ind] = rate_RRneuron_dr.copy()

                # # trial order re-randomization
                # for trial_type_ind in range(num_trial_types):
                #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                # repeat calculating similarity matrices

                # n_neurons x n_stimuli 2D matrix sampling

                # if ind == 3:
                #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
                tt_pairs = list(combinations(range(min_num_trials), 2))
                n_sampling = np.min([len(tt_pairs), 10000])
                # n_sampling = len(tt_pairs)

                list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
                
                count = 0
                for sampling_ind in range(n_sampling):
                    
                    rate_sampled_trials1 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][0]]).copy() 
                    rate_sampled_trials2 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][1]]).copy()

                    # calculate similarity matrix
                    RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) 

                    # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) 
                    list_RSM[sampling_ind] = RSM.copy()
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f'count: {count}')

                RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
                list_RSM_mean_RRneuron[ind] = RSM_mean.copy()

        # Spearman correlation across session pairs
        sess_pairs = list(combinations(range(num_sess), 2))
        list_corr_sesspair_asis = np.zeros((len(sess_pairs), 3)) 
        list_corr_sesspair = np.zeros((len(sess_pairs), 3)) 
        for pair_ind, pair in enumerate(sess_pairs):
            if slope_ind == 0:
                RSM_mean_neu1 = list_RSM_mean_asis[pair[0]].copy()
                RSM_mean_neu2 = list_RSM_mean_asis[pair[1]].copy()

                list_corr_sesspair_asis[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_sesspair_asis[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_sesspair_asis[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())                
            
            RSM_mean_neu1 = list_RSM_mean_RRneuron[pair[0]].copy()
            RSM_mean_neu2 = list_RSM_mean_RRneuron[pair[1]].copy()

            list_corr_sesspair[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_sesspair[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_sesspair[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

        list_RSM_mean_asis_HVA[area] = list_RSM_mean_asis.copy()
        list_corr_sesspair_asis_HVA[area] = list_corr_sesspair_asis.copy()

        list_RSM_mean_RRneuron_HVA[area] = list_RSM_mean_RRneuron.copy()
        list_rate_RRneuron_dr_HVA[area] = list_rate_RRneuron_dr.copy()
        list_corr_sesspair_HVA[area] = list_corr_sesspair.copy()
    
    # Save into a file
    filename = 'RSM_corr_sesspair_ABO_HVA_allneu_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis_HVA', 'list_corr_sesspair_asis_HVA', 'list_rate_RRneuron_dr_HVA', 'list_RSM_mean_RRneuron_HVA', 'list_corr_sesspair_HVA'],
                     'list_RSM_mean_asis_HVA': list_RSM_mean_asis_HVA, 'list_corr_sesspair_asis_HVA': list_corr_sesspair_asis_HVA,
                     'list_rate_RRneuron_dr_HVA': list_rate_RRneuron_dr_HVA, 'list_RSM_mean_RRneuron_HVA': list_RSM_mean_RRneuron_HVA, 'list_corr_sesspair_HVA': list_corr_sesspair_HVA}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA within sessions (ABO, RRneuron)

def RSA_withinsess_ABO(slope_ind, target_slope, similarity_type='cos_sim'):
    
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # number of trial sampling for each stimulus
    num_trials = 50
    num_trial_types = 119
    num_sess = 32
    n_neu_sampling = 10

    print(f'target slope {target_slope:.1f}')

    # Iterate over all sessions
    np.random.seed(0) # match neuron partitioning
    # random.seed(0)

    list_corr_withinsess_asis2 = np.full((num_sess, n_neu_sampling, 3), np.nan) 
    list_corr_withinsess2 = np.full((num_sess, n_neu_sampling, 3), np.nan) 

    for sess_ind in range(num_sess):

        print(f'sess_ind: {sess_ind}')

        rate = list_rate_all[sess_ind].copy()
        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Multiply by delta t to convert to spike counts
        rate = rate * 0.25

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) 
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # convert to 3D response matrix
        min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
            list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        rate = np.stack(list_rate_tt, axis=2)
        rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
    
        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
        
        list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

        # trial shuffling
        rate_shuf = np.zeros_like(rate_sorted)
        for neu_ind in range(rate_sorted.shape[0]):
            shuf_inds = np.random.permutation(rate_sorted.shape[2])
            rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() 
        # rate_sorted = rate_shuf.copy()

        # trial order re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        # calculate target variance
        var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed 
        offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) 

        var_rs_noisy = \
            pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
        var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

        # Compute changed residual and add back to the mean            
        rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
            * np.sqrt(var_rs_noisy)
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!      

        # # trial order re-randomization
        # for trial_type_ind in range(num_trial_types):
        #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)] 

        # Iterate over neuron partitionings
        for neu_sample_ind in range(n_neu_sampling):
            print(f'neu_sample_ind = {neu_sample_ind}')
            
            # Partition neurons
            neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
            neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 partitioning
            neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
            if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
                neu_div_inds2 = neu_div_inds2[:-1].copy()

            # as-is

            if slope_ind == 0:

                # repeat calculating similarity matrices
                
                # n_neurons x n_stimuli 2D matrix sampling
                
                tt_pairs = list(combinations(range(min_num_trials), 2))
                # random.shuffle(tt_pairs) 
                n_sampling = np.min([len(tt_pairs), 10000])
                # n_sampling = len(tt_pairs)

                list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                count = 0
                for sampling_ind in range(n_sampling):
                    
                    rate_sampled_trials1_1 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() 
                    rate_sampled_trials1_2 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                    rate_sampled_trials2_1 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() 
                    rate_sampled_trials2_2 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                    RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) 
                    RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) 

                    list_RSM_neu1[sampling_ind] = RSM1.copy()
                    list_RSM_neu2[sampling_ind] = RSM2.copy()

                RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # Average across trial samplings
                RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
        
            # Change slope

            # repeat calculating similarity matrices
            
            # n_neurons x n_stimuli 2D matrix sampling
            
            tt_pairs = list(combinations(range(min_num_trials), 2))
            # random.shuffle(tt_pairs) 
            n_sampling = np.min([len(tt_pairs), 10000])
            # n_sampling = len(tt_pairs)

            list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
            list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

            count = 0
            for sampling_ind in range(n_sampling):
                
                rate_sampled_trials1_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() 
                rate_sampled_trials1_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                rate_sampled_trials2_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() 
                rate_sampled_trials2_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) 
                RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) 

                # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) 
                list_RSM_neu1[sampling_ind] = RSM1.copy()
                list_RSM_neu2[sampling_ind] = RSM2.copy()

                count += 1
                if count % 1000 == 0:
                    print(f'count: {count}')

            RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # Average across trial samplings
            RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
            list_corr_withinsess2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_withinsess2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_withinsess2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_withinsess_ABO_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis2', 'list_corr_withinsess2'], \
                    'list_corr_withinsess_asis2': list_corr_withinsess_asis2, 'list_corr_withinsess2': list_corr_withinsess2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# RSA within sessions (ABO, RRneuron)

def RSA_withinsess_ABO_HVA(slope_ind, target_slope, similarity_type='cos_sim'):
    
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # number of trial sampling for each stimulus
    num_trials = 50
    num_trial_types = 119
    num_sess = 32
    n_neu_sampling = 10

    print(f'target slope {target_slope:.1f}')

    np.random.seed(0) # match neuron partitioning
    # random.seed(0)

    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']

    list_corr_withinsess_asis_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_corr_withinsess_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    for area_ind, area in enumerate(list_HVA_names):
                
        # Iterate over all sessions

        list_corr_withinsess_asis2 = np.full((num_sess, n_neu_sampling, 3), np.nan) 
        list_corr_withinsess2 = np.full((num_sess, n_neu_sampling, 3), np.nan) 

        for sess_ind in range(num_sess):

            # print(f'area: {area}, sess_ind: {sess_ind}')

            rate = list_rate_all_HVA[area][sess_ind].copy()
            if np.any(rate) == True: # if neurons exist

                # rate_sorted = rate.sort_index(axis=1)
                stm = rate.columns.copy()

                # Multiply by delta t to convert to spike counts
                rate = rate * 0.25

                # Create a counting dictionary for each stimulus
                all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) 
                stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
                
                # convert to 3D response matrix
                min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

                list_rate_tt = [None] * num_trial_types
                for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
                    list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

                rate = np.stack(list_rate_tt, axis=2)
                rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
            
                rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
                rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
                    np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
                
                list_slopes_dr = list_slopes_all_an_loglog_HVA[area][sess_ind].copy()

                # trial shuffling
                rate_shuf = np.zeros_like(rate_sorted)
                for neu_ind in range(rate_sorted.shape[0]):
                    shuf_inds = np.random.permutation(rate_sorted.shape[2])
                    rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() 
                # rate_sorted = rate_shuf.copy()

                # trial order re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

                # calculate target variance
                var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

                # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
                # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed 
                offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) 

                var_rs_noisy = \
                    pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                        / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
                var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

                # Compute changed residual and add back to the mean            
                rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
                # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                    * np.sqrt(var_rs_noisy)
                # print(rate_resid_RRneuron_dr)
                rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
                rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!      

                # # trial order re-randomization
                # for trial_type_ind in range(num_trial_types):
                #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)] 

                # Iterate over neuron partitionings
                for neu_sample_ind in range(n_neu_sampling):
                    print(f'area: {area}, sess_ind: {sess_ind}, neu_sample_ind = {neu_sample_ind}')
                    
                    # Partition neurons
                    neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
                    neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 partitioning
                    neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
                    if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
                        neu_div_inds2 = neu_div_inds2[:-1].copy()

                    # as-is

                    if slope_ind == 0:

                        # repeat calculating similarity matrices
                        
                        # n_neurons x n_stimuli 2D matrix sampling
                        
                        tt_pairs = list(combinations(range(min_num_trials), 2))
                        # random.shuffle(tt_pairs) 
                        n_sampling = np.min([len(tt_pairs), 10000])
                        # n_sampling = len(tt_pairs)

                        list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                        list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                        count = 0
                        for sampling_ind in range(n_sampling):
                            
                            rate_sampled_trials1_1 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() 
                            rate_sampled_trials1_2 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                            rate_sampled_trials2_1 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() 
                            rate_sampled_trials2_2 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                            RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) 
                            RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) 

                            list_RSM_neu1[sampling_ind] = RSM1.copy()
                            list_RSM_neu2[sampling_ind] = RSM2.copy()

                        RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # Average across trial samplings
                        RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                        list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
                        bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                        list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                        list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
                
                    # Change slope

                    # repeat calculating similarity matrices
                    
                    # n_neurons x n_stimuli 2D matrix sampling
                    
                    tt_pairs = list(combinations(range(min_num_trials), 2))
                    # random.shuffle(tt_pairs) 
                    n_sampling = np.min([len(tt_pairs), 10000])
                    # n_sampling = len(tt_pairs)

                    list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                    list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                    count = 0
                    for sampling_ind in range(n_sampling):
                        
                        rate_sampled_trials1_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() 
                        rate_sampled_trials1_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                        rate_sampled_trials2_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() 
                        rate_sampled_trials2_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                        RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) 
                        RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) 

                        # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) 
                        list_RSM_neu1[sampling_ind] = RSM1.copy()
                        list_RSM_neu2[sampling_ind] = RSM2.copy()

                        count += 1
                        if count % 1000 == 0:
                            print(f'count: {count}')

                    RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # Average across trial samplings
                    RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                    list_corr_withinsess2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic 
                    bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                    list_corr_withinsess2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                    list_corr_withinsess2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

        list_corr_withinsess_asis_HVA[area] = list_corr_withinsess_asis2.copy()
        list_corr_withinsess_HVA[area] = list_corr_withinsess2.copy()

    # Save into a file
    filename = 'RSM_corr_withinsess_ABO_HVA_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis_HVA', 'list_corr_withinsess_HVA'], \
                    'list_corr_withinsess_asis_HVA': list_corr_withinsess_asis_HVA, 'list_corr_withinsess_HVA': list_corr_withinsess_HVA}, f)
        
    print("Ended Process", c_proc.name)

# %%
# decoding (ABO)
def decode_ABO(sess_ind, decoder_type):

    ''' decoder_type is SVM, logit,, RF, kNN '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        num_sess = 32
        num_trial_types = 119
        num_trials = 50
        n_splits = 10
        t_win = 0.25

        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

        np.random.seed(0)

        all_stimuli = np.arange(-1, 118, 1).astype(int) # include grayscreen trials
        probe_stimuli = np.array([5, 12, 24, 34, 36, 44, 47, 78, 83, 87, 104, 111, 114, 115])
        # train_stimuli = all_stimuli[~np.isin(all_stimuli, probe_stimuli)].copy()
        train_stimuli = all_stimuli.copy()

        rate = list_rate_all[sess_ind].copy()
        
        # print(f'sess_ind: {sess_ind}')

        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Multiply by delta t to convert to spike counts
        rate = rate * 0.25

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) 
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # convert to 3D response matrix
        min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(all_stimuli):
            list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        rate = np.stack(list_rate_tt, axis=2)
        rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x min_num_trials

        # trial shuffling
        rate_shuf = np.zeros_like(rate_sorted)
        for neu_ind in range(rate_sorted.shape[0]):
            shuf_inds = np.random.permutation(rate_sorted.shape[2])
            rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() 
        # rate_sorted = rate_shuf.copy()

        # Compute mean & variance for each stimulus
        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)

        list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

        # trial order re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]
        
        # decoding cross-validation (as-is)
        kfold = KFold(n_splits=n_splits)
        stkfold = StratifiedKFold(n_splits=n_splits)

        # Re-convert to 2D response matrix
        label_train = np.repeat(all_stimuli, min_num_trials)
        rate_train = pd.DataFrame(rate_sorted.reshape(rate_sorted.shape[0], -1), columns=label_train)
        
        # decoding cross-validation (as-is)
        kfold = KFold(n_splits=n_splits)
        stkfold = StratifiedKFold(n_splits=n_splits)

        list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
        list_accuracy = np.full(n_splits, np.nan)

        if decoder_type in ['SVM', 'logit', 'RF', 'kNN']:

            for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
                X_train, X_test = rate_train.T.iloc[train_index].copy(), rate_train.T.iloc[test_index].copy() # train, test data/label
                y_train, y_test = label_train[train_index].copy(), label_train[test_index].copy()

                mean_ = X_train.mean(axis=0)
                X_train = X_train.sub(mean_, axis=1) # train data mean centering
                X_test = X_test.sub(mean_, axis=1)

                if decoder_type == 'SVM':
                    clf = svm.SVC(kernel='linear', max_iter=-1)
                elif decoder_type == 'logit':
                    clf = logit(max_iter=100) # default: L2 regularization, lbfgs solver, C=1
                elif decoder_type == 'RF':
                    clf = rf()
                elif decoder_type == 'kNN':
                    clf = KNeighborsClassifier(n_neighbors=30)  
                                
                clf.fit(X_train, y_train) # SVC fitting to train data
                
                y_test_pred = clf.predict(X_test) # predicted label for test data

                # record normalized test confusion matrix/test accuracy
                test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
                test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                list_confusion_test[split_ind] = test_confusion_matrix.copy()

                accuracy = accuracy_score(y_test, y_test_pred)
                list_accuracy[split_ind] = accuracy
                # print(accuracy)
            
                # print(f'split_ind {split_ind}')

        # calculate cross-validation average test confusion matrix/test accuracy
        mean_confusion_test_asis = sum(list_confusion_test) / n_splits
        mean_confusion_test_asis = pd.DataFrame(mean_confusion_test_asis, columns=train_stimuli, index=train_stimuli).fillna(0)
        # print(mean_confusion_test.round(3))
        
        mean_accuracy_asis = np.mean(list_accuracy)
        # print(round(mean_accuracy, ndigits=3))

        # Change slope
        list_mean_confusion_test_RRneuron = np.full((len(list_target_slopes), len(train_stimuli), len(train_stimuli)), np.nan)
        list_mean_accuracy_RRneuron = np.full(len(list_target_slopes), np.nan)

        for slope_ind, target_slope in enumerate(list_target_slopes):
            start_time = time()

            print(f'sess_ind: {sess_ind}, target slope {target_slope:.1f}')
                                
            # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
            rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
            rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

            # calculate target variance
            var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

            # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed 
            offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) 

            var_rs_noisy = \
                pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                    / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
            var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

            # Compute changed residual and add back to the mean
            rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                * np.sqrt(var_rs_noisy)
            # print(rate_resid_RRneuron_dr)
            rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!        

            # # trial order re-randomization
            # for trial_type_ind in range(num_trial_types):
            #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]
            
            # decoding cross-validation (RRneuron)  

            # Re-convert to 2D response matrix
            label_train_RRneuron = np.repeat(all_stimuli, min_num_trials)
            rate_train_RRneuron = pd.DataFrame(rate_RRneuron_dr.reshape(rate_RRneuron_dr.shape[0], -1), columns=label_train_RRneuron)
            
            # decoding cross-validation (as-is)
            kfold = KFold(n_splits=n_splits)
            stkfold = StratifiedKFold(n_splits=n_splits)

            list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
            list_accuracy = np.full(n_splits, np.nan)

            if decoder_type in ['SVM', 'logit', 'RF', 'kNN']:

                for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
                    X_train, X_test = rate_train_RRneuron.T.iloc[train_index].copy(), rate_train_RRneuron.T.iloc[test_index].copy() # train, test data/label
                    y_train, y_test = label_train_RRneuron[train_index].copy(), label_train_RRneuron[test_index].copy()

                    mean_ = X_train.mean(axis=0)
                    X_train = X_train.sub(mean_, axis=1) # train data mean centering
                    X_test = X_test.sub(mean_, axis=1)

                    if decoder_type == 'SVM':
                        clf = svm.SVC(kernel='linear', max_iter=-1)
                    elif decoder_type == 'logit':
                        clf = logit(max_iter=100) # default: L2 regularization, lbfgs solver, C=1
                    elif decoder_type == 'RF':
                        clf = rf()
                    elif decoder_type == 'kNN':
                        clf = KNeighborsClassifier(n_neighbors=30) 
                                        
                    clf.fit(X_train, y_train) # SVC fitting to train data
                    
                    y_test_pred = clf.predict(X_test) # predicted label for test data

                    # record normalized test confusion matrix/test accuracy
                    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
                    test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                    list_confusion_test[split_ind] = test_confusion_matrix.copy()

                    accuracy = accuracy_score(y_test, y_test_pred)
                    list_accuracy[split_ind] = accuracy
                    # print(accuracy)
                                                
                    # print(f'split_ind {split_ind}')

            # calculate cross-validation average test confusion matrix/test accuracy
            mean_confusion_test = sum(list_confusion_test) / n_splits
            mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
            # print(mean_confusion_test_Bayes.round(3))
            list_mean_confusion_test_RRneuron[slope_ind] = mean_confusion_test.copy()
            
            mean_accuracy = np.mean(list_accuracy)
            # print(round(mean_accuracy, ndigits=3))
            list_mean_accuracy_RRneuron[slope_ind] = mean_accuracy

            print(f'sess_ind: {sess_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = decoder_type + '_decoding_ABO_allstim_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['mean_confusion_test_asis', 'mean_accuracy_asis', 'list_mean_confusion_test_RRneuron', 'list_mean_accuracy_RRneuron'],
                     'mean_confusion_test_asis': mean_confusion_test_asis, 'mean_accuracy_asis': mean_accuracy_asis,
                     'list_mean_confusion_test_RRneuron': list_mean_confusion_test_RRneuron, 'list_mean_accuracy_RRneuron': list_mean_accuracy_RRneuron}, f)
                
    print("Ended Process", c_proc.name)

# %%
# decoding (ABO, HVA)
def decode_ABO_HVA(slope_ind, target_slope, decoder_type='SVM'):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_sess = 32
    num_trial_types = 119
    num_trials = 50
    n_splits = 10
    t_win = 0.25

    np.random.seed(0) # match neuron partitioning.

    all_stimuli = np.arange(-1, 118, 1).astype(int) # include grayscreen trials
    probe_stimuli = np.array([5, 12, 24, 34, 36, 44, 47, 78, 83, 87, 104, 111, 114, 115])
    # train_stimuli = all_stimuli[~np.isin(all_stimuli, probe_stimuli)].copy()
    train_stimuli = all_stimuli.copy()

    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']

    list_mean_confusion_test_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_mean_accuracy_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}

    list_mean_confusion_test_RRneuron_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_mean_accuracy_RRneuron_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}

    for area_ind, area in enumerate(list_HVA_names):

        # Iterate over all sessions

        list_mean_confusion_test = []
        list_mean_accuracy = []

        list_mean_confusion_test_RRneuron = []
        list_mean_accuracy_RRneuron = []

        for sess_ind in range(num_sess):
            # print(f'area {area}, sess_ind: {sess_ind}')

            rate = list_rate_all_HVA[area][sess_ind].copy()

            if np.any(rate) == True: # neuron이 있을 때
                stm = rate.columns.copy()

                # Multiply by delta t to convert to spike counts
                rate = rate * 0.25

                # Create a counting dictionary for each stimulus
                all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) 
                stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
                
                # convert to 3D response matrix
                min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

                list_rate_tt = [None] * num_trial_types
                for trial_type_ind, trial_type in enumerate(all_stimuli):
                    list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

                rate = np.stack(list_rate_tt, axis=2)
                rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x min_num_trials

                # Compute mean & variance for each stimulus
                rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
                rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
                    np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)

                list_slopes_dr = list_slopes_all_an_loglog_HVA[area][sess_ind].copy()

                # trial order re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]
                
                # decoding cross-validation (as-is)
                kfold = KFold(n_splits=n_splits)
                stkfold = StratifiedKFold(n_splits=n_splits)
                
                if slope_ind == 0:

                    # Re-convert to 2D response matrix
                    label_train = np.repeat(all_stimuli, min_num_trials)
                    rate_train = pd.DataFrame(rate_sorted.reshape(rate_sorted.shape[0], -1), columns=label_train)
                    
                    # decoding cross-validation (as-is)
                    kfold = KFold(n_splits=n_splits)
                    stkfold = StratifiedKFold(n_splits=n_splits)

                    list_confusion_test = []
                    list_accuracy = []

                    if decoder_type == 'SVM':
                        for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
                            X_train, X_test = rate_train.T.iloc[train_index].copy(), rate_train.T.iloc[test_index].copy() # train, test data/label
                            y_train, y_test = label_train[train_index].copy(), label_train[test_index].copy()

                            mean_ = X_train.mean(axis=0)
                            X_train = X_train.sub(mean_, axis=1) # train data mean centering
                            X_test = X_test.sub(mean_, axis=1)

                            clf_SVC = svm.SVC(kernel='linear')
                            # clf_SVC = svm.SVC(kernel='poly', degree=3)
                            # clf_SVC = svm.SVC(kernel='rbf')
                            
                            clf_SVC.fit(X_train, y_train) # SVC fitting to train data
                            
                            y_test_pred_SVC = clf_SVC.predict(X_test) # predicted label for test data

                            # record normalized test confusion matrix/test accuracy
                            test_confusion_matrix = confusion_matrix(y_test, y_test_pred_SVC)
                            test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                            list_confusion_test.append(test_confusion_matrix)

                            accuracy = accuracy_score(y_test, y_test_pred_SVC)
                            list_accuracy.append(accuracy)
                            # print(accuracy)
                        
                            # print(f'split_ind {split_ind}')

                    # calculate cross-validation average test confusion matrix/test accuracy
                    mean_confusion_test = sum(list_confusion_test) / n_splits
                    mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
                    # print(mean_confusion_test.round(3))
                    list_mean_confusion_test.append(mean_confusion_test)
                    
                    mean_accuracy = np.mean(list_accuracy)
                    # print(round(mean_accuracy, ndigits=3))
                    list_mean_accuracy.append(mean_accuracy)

                # Change slope

                print(f'target slope {target_slope:.1f}, area {area}, sess_ind: {sess_ind}')
                                    
                # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

                # calculate target variance
                var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

                # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
                # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed 
                offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) 

                var_rs_noisy = \
                    pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                        / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
                var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

                # Compute changed residual and add back to the mean
                rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
                # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                    * np.sqrt(var_rs_noisy)
                # print(rate_resid_RRneuron_dr)
                rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
                rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!        

                # # trial order re-randomization
                # for trial_type_ind in range(num_trial_types):
                #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                # decoding cross-validation (RRneuron)  

                # Re-convert to 2D response matrix
                label_train_RRneuron = np.repeat(all_stimuli, min_num_trials)
                rate_train_RRneuron = pd.DataFrame(rate_RRneuron_dr.reshape(rate_RRneuron_dr.shape[0], -1), columns=label_train_RRneuron)
                
                # decoding cross-validation (as-is)
                kfold = KFold(n_splits=n_splits)
                stkfold = StratifiedKFold(n_splits=n_splits)

                list_confusion_test = []
                list_accuracy = []

                if decoder_type == 'SVM':
                    for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
                        X_train, X_test = rate_train_RRneuron.T.iloc[train_index].copy(), rate_train_RRneuron.T.iloc[test_index].copy() # train, test data/label
                        y_train, y_test = label_train_RRneuron[train_index].copy(), label_train_RRneuron[test_index].copy()

                        mean_ = X_train.mean(axis=0)
                        X_train = X_train.sub(mean_, axis=1) # train data mean centering
                        X_test = X_test.sub(mean_, axis=1)

                        clf_SVC = svm.SVC(kernel='linear')
                        # clf_SVC = svm.SVC(kernel='poly', degree=3)
                        # clf_SVC = svm.SVC(kernel='rbf')
                        
                        clf_SVC.fit(X_train, y_train) # SVC fitting to train data
                        
                        y_test_pred_SVC = clf_SVC.predict(X_test) # predicted label for test data

                        # record normalized test confusion matrix/test accuracy
                        test_confusion_matrix = confusion_matrix(y_test, y_test_pred_SVC)
                        test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                        list_confusion_test.append(test_confusion_matrix)

                        accuracy = accuracy_score(y_test, y_test_pred_SVC)
                        list_accuracy.append(accuracy)
                        # print(accuracy)
                    
                        # print(f'split_ind {split_ind}')

                # calculate cross-validation average test confusion matrix/test accuracy
                mean_confusion_test = sum(list_confusion_test) / n_splits
                mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
                # print(mean_confusion_test.round(3))
                list_mean_confusion_test_RRneuron.append(mean_confusion_test)
                
                mean_accuracy = np.mean(list_accuracy)
                # print(round(mean_accuracy, ndigits=3))
                list_mean_accuracy_RRneuron.append(mean_accuracy)

        list_mean_confusion_test_HVA[area] = list_mean_confusion_test.copy()
        list_mean_accuracy_HVA[area] = list_mean_accuracy.copy()

        list_mean_confusion_test_RRneuron_HVA[area] = list_mean_confusion_test_RRneuron.copy()
        list_mean_accuracy_RRneuron_HVA[area] = list_mean_accuracy_RRneuron.copy()

    # Save into a file
    filename = decoder_type + '_decoding_ABO_HVA_' + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_mean_confusion_test_HVA', 'list_mean_accuracy_HVA', 'list_mean_confusion_test_RRneuron_HVA', 'list_mean_accuracy_RRneuron_HVA'],
                     'list_mean_confusion_test_HVA': list_mean_confusion_test_HVA, 'list_mean_accuracy_HVA': list_mean_accuracy_HVA,
                     'list_mean_confusion_test_RRneuron_HVA': list_mean_confusion_test_RRneuron_HVA, 'list_mean_accuracy_RRneuron_HVA': list_mean_accuracy_RRneuron_HVA}, f)

    print("Ended Process", c_proc.name)

# %%
# loading variables

# ABO Neuropixels
with open('resp_matrix_ep_RS_all_32sess_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_RS = resp_matrix_ep_RS_all['list_rate_RS'].copy()
    list_rate_RS_dr = resp_matrix_ep_RS_all['list_rate_RS_dr'].copy()
    list_rate_all = resp_matrix_ep_RS_all['list_rate_all'].copy()
    list_rate_all_dr = resp_matrix_ep_RS_all['list_rate_all_dr'].copy()
    list_slopes_RS_an_loglog = resp_matrix_ep_RS_all['list_slopes_RS_an_loglog'].copy()
    list_slopes_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_all_an_loglog'].copy()

# ABO higher visual areas
with open('resp_matrix_ep_HVA_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_HVA_allensdk = pickle.load(f)

    list_rate_all_HVA = dc(resp_matrix_ep_HVA_allensdk['list_rate_all_HVA'])
    list_slopes_all_an_loglog_HVA = dc(resp_matrix_ep_HVA_allensdk['list_slopes_all_an_loglog_HVA'])
    list_empty_sess2 = dc(resp_matrix_ep_HVA_allensdk['list_empty_sess2'])

# %%
# multiprocessing
list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
num_sess = 32

# RSA across session pairs
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(RSA_across_sesspairs_ABO, list_inputs)

# RSA across session pairs
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(RSA_across_sesspairs_ABO_HVA, list_inputs)

# RSA within sessions
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(RSA_withinsess_ABO, list_inputs)

# RSA within sessions
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(RSA_withinsess_ABO_HVA, list_inputs)

# decoding
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[sess_ind, 'SVM'] for sess_ind in range(num_sess)]
        
        pool.starmap(decode_ABO, list_inputs)

# decoding
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[slope_ind, target_slope, 'SVM'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(decode_ABO_HVA, list_inputs)