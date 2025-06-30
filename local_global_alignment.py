# %%
import mat73
import h5py
import hdf5storage as st
# from pymatreader import read_mat
import pickle
import os

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
from copy import deepcopy
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
# Function to compute cosine similarity
def cos_sim(x, y):
    # x and y are 1D vectors

    # dot_xy = np.dot(x, y)
    # norm_x, norm_y = np.linalg.norm(x.astype(np.float32)), np.linalg.norm(y.astype(np.float32))

    # cos_sim = dot_xy / (norm_x * norm_y)

    return np.dot(x, y) / (np.linalg.norm(x.astype(np.float32)) * np.linalg.norm(y.astype(np.float32)))

# %%
# ABO PC1 analysis (PC1 of each stimulus vs. local/global PC1)

def compute_cos_sim_pc1_adj_ABO(slope_ind, target_slope, adjacency_type='geodesic'):
    
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_sess = 32
    num_trial_types = 119

    # Iterate over all sessions

    list_cos_sim_pc1_adj2 = np.zeros((num_sess, num_trial_types))
    list_cos_sim_pc1_adj_RRneuron2 = np.zeros((num_sess, num_trial_types))
    list_cos_sim_pc1_ori2 = np.zeros((num_sess, num_trial_types))
    list_cos_sim_pc1_ori_RRneuron2 = np.zeros((num_sess, num_trial_types))
    list_cos_sim_pc1_global2 = np.zeros((num_sess, num_trial_types))
    list_cos_sim_pc1_global_RRneuron2 = np.zeros((num_sess, num_trial_types))
    for sess_ind, rate in enumerate(list_rate_all):
        print(f'session index: {sess_ind}')
        
        rate_sorted = rate.sort_index(axis=1)
        stm = rate_sorted.columns.copy()

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
            
        # concatenate centroids of all stimuli
        rate_plus_mean = pd.concat([rate_sorted, rate_sorted_mean_coll], axis=1)

        # Compute geodesic distance matrix
        n_components = 1 # target number of dimensions
        n_neighbors = 5 # number of neighbors

        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        
        isomap.fit(rate_plus_mean.T)
        mean_dist_mat_asis = isomap.dist_matrix_[rate_sorted.shape[1]:, rate_sorted.shape[1]:].copy() # inter-centroid geodesic distance matrix

        # Compute local/global alignment
        n_components = 1
        pca = PCA(n_components=n_components)

        if slope_ind == 0:

            list_cos_sim_pc1_adj = np.zeros(rate_sorted_mean_coll.shape[1])
            list_cos_sim_pc1_ori = np.zeros(rate_sorted_mean_coll.shape[1])
            list_cos_sim_pc1_global = np.zeros(rate_sorted_mean_coll.shape[1])
            for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):
                print(f'trial_type_ind = {trial_type_ind}')

                rate_tt = rate_sorted.loc[:, trial_type].copy()
                rate_tt_pca = pca.fit_transform(rate_tt.T).T
                pc_tt = pca.components_.copy() 

                # Determine the nearest neighbor stimulus manifold

                bool_not_tt = rate_sorted_mean_coll.columns != trial_type

                adj_tt_ind = np.argmin(mean_dist_mat_asis[trial_type_ind, bool_not_tt]) # exclude current stimulus
                if adj_tt_ind >= trial_type_ind:
                    adj_tt_ind = adj_tt_ind + 1 # to original index before excluding the stimulus
                adj_tt = rate_sorted_mean_coll.columns[adj_tt_ind]

                # compute alignment with neighboring stimulus PC1
                rate_adjtt = rate_sorted.loc[:, adj_tt].copy()
                rate_adjtt_pca = pca.fit_transform(rate_adjtt.T).T
                pc_adjtt = pca.components_.copy() 
                list_cos_sim_pc1_adj[trial_type_ind] = np.abs(cos_sim(pc_tt[0], pc_adjtt[0])) 

                mean_vector = rate_sorted_mean_coll.iloc[:, trial_type_ind].copy() # mean vector referencing the origin of state space
                list_cos_sim_pc1_ori[trial_type_ind] = np.abs(cos_sim(pc_tt[0], mean_vector)) 

                # compute alignment with global PC1
                list_not_tt = rate_sorted_mean_coll.columns[bool_not_tt].copy()
                rate_pca = pca.fit_transform(rate_sorted.loc[:, list_not_tt].T).T # all stimuli excluding current stimulus
                pc1_global = pca.components_[0].copy() 

                list_cos_sim_pc1_global[trial_type_ind] = np.abs(cos_sim(pc_tt[0], pc1_global))

            list_cos_sim_pc1_adj2[sess_ind] = list_cos_sim_pc1_adj.copy()
            list_cos_sim_pc1_ori2[sess_ind] = list_cos_sim_pc1_ori.copy()
            list_cos_sim_pc1_global2[sess_ind] = list_cos_sim_pc1_global.copy()
        
        print(f'target_slope = {target_slope:.1f}')

        # Change slope

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        # calculate target variance
        var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                columns=rate_sorted_var_coll.columns) 
        for trial_type in rate_sorted_var_coll.columns:
            var_estim_dr.loc[:, trial_type] = \
                np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
        # var_estim_dr = np.repeat(var_estim_dr, all_stm_counts, axis=1) 
        # print(var_estim_dr)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed 
        offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr.iloc[1, :]) 

        var_rs_noisy = \
            pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
                .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # collapsed
        var_rs_noisy = np.repeat(np.array(var_rs_noisy), all_stm_counts, axis=1) 

        # Compute changed residual and add back to the mean            
        rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            .mul(np.sqrt(var_rs_noisy))
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # convert NaN to 0!    

        # Compute mean and variance of slope-changed data
        rate_mean_RRneuron_coll, rate_var_RRneuron_coll = \
            compute_mean_var_trial_collapse(stm_cnt_dict, rate_RRneuron_dr)
        # FF_RRneuron = rate_var_RRneuron_dr.div(rate_mean_RRneuron_dr)
        # print(FF_RRneuron)
        # print(rate_var_RRneuron_dr)

        # Compute local/global alignment
        list_cos_sim_pc1_adj_RRneuron = np.zeros(rate_mean_RRneuron_coll.shape[1])
        list_cos_sim_pc1_ori_RRneuron = np.zeros(rate_mean_RRneuron_coll.shape[1])
        list_cos_sim_pc1_global_RRneuron = np.zeros(rate_mean_RRneuron_coll.shape[1])
        for trial_type_ind, trial_type in enumerate(rate_mean_RRneuron_coll.columns):
            rate_tt_RRneuron = rate_RRneuron_dr.loc[:, trial_type].copy()
            rate_tt_RRneuron_pca = pca.fit_transform(rate_tt_RRneuron.T).T
            pc_tt_RRneuron = pca.components_.copy() 

            # Determine the nearest neighbor stimulus manifold

            bool_not_tt = rate_mean_RRneuron_coll.columns != trial_type

            adj_tt_ind = np.argmin(mean_dist_mat_asis[trial_type_ind, bool_not_tt]) # exclude current stimulus            
            if adj_tt_ind >= trial_type_ind:
                adj_tt_ind = adj_tt_ind + 1 # to original index before excluding the stimulus 
            adj_tt = rate_sorted_mean_coll.columns[adj_tt_ind]

            # compute alignment with neighboring stimulus PC1
            rate_adjtt_RRneuron = rate_RRneuron_dr.loc[:, adj_tt].copy()
            rate_adjtt_RRneuron_pca = pca.fit_transform(rate_adjtt_RRneuron.T).T
            pc_adjtt_RRneuron = pca.components_.copy() 
            list_cos_sim_pc1_adj_RRneuron[trial_type_ind] = np.abs(cos_sim(pc_tt_RRneuron[0], pc_adjtt_RRneuron[0])) 

            mean_vector_RRneuron = rate_mean_RRneuron_coll.iloc[:, trial_type_ind].copy()
            list_cos_sim_pc1_ori_RRneuron[trial_type_ind] = np.abs(cos_sim(pc_tt_RRneuron[0], mean_vector_RRneuron)) 

            # compute alignment with global PC1
            list_not_tt = rate_mean_RRneuron_coll.columns[bool_not_tt].copy()
            rate_RRneuron_pca = pca.fit_transform(rate_RRneuron_dr.loc[:, list_not_tt].T).T
            pc1_global_RRneuron = pca.components_[0].copy() 

            list_cos_sim_pc1_global_RRneuron[trial_type_ind] = np.abs(cos_sim(pc_tt_RRneuron[0], pc1_global_RRneuron))

        list_cos_sim_pc1_adj_RRneuron2[sess_ind] = list_cos_sim_pc1_adj_RRneuron.copy()
        list_cos_sim_pc1_ori_RRneuron2[sess_ind] = list_cos_sim_pc1_ori_RRneuron.copy()
        list_cos_sim_pc1_global_RRneuron2[sess_ind] = list_cos_sim_pc1_global_RRneuron.copy()

    # Save into a file
    filename = 'D:\\Users\\USER\\Shin Lab\\code\\align_pc1_ABO_' + adjacency_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_cos_sim_pc1_adj2', 'list_cos_sim_pc1_ori2', 'list_cos_sim_pc1_global2',
                                        'list_cos_sim_pc1_adj_RRneuron2', 'list_cos_sim_pc1_ori_RRneuron2', 'list_cos_sim_pc1_global_RRneuron2'],
                                        'list_cos_sim_pc1_adj2': list_cos_sim_pc1_adj2, 'list_cos_sim_pc1_ori2': list_cos_sim_pc1_ori2, 'list_cos_sim_pc1_global2': list_cos_sim_pc1_global2,
                                        'list_cos_sim_pc1_adj_RRneuron2': list_cos_sim_pc1_adj_RRneuron2, 'list_cos_sim_pc1_ori_RRneuron2': list_cos_sim_pc1_ori_RRneuron2,
                                        'list_cos_sim_pc1_global_RRneuron2': list_cos_sim_pc1_global_RRneuron2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# loading variables

# openscope
with open('SVM_prerequisite_variables.pickle', 'rb') as f:
    SVM_prerequisite_variables = pickle.load(f)
    
    list_rate_w1 = SVM_prerequisite_variables['list_rate_w1'].copy()
    list_stm_w1 = SVM_prerequisite_variables['list_stm_w1'].copy()
    list_neu_loc = SVM_prerequisite_variables['list_neu_loc'].copy()
    list_wfdur = SVM_prerequisite_variables['list_wfdur'].copy()
    list_slopes_an_loglog_12 = SVM_prerequisite_variables['list_slopes_an_loglog_12'].copy() # high repeat trial type

# ABO Neuropixels
with open('resp_matrix_ep_RS_all_32sess_allensdk.pickle', 'rb') as f:
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
list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

# alignment PC1
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[slope_ind, target_slope, 'geodesic'] for slope_ind, target_slope in enumerate(list_target_slopes) if slope_ind == 0]
        
        pool.starmap(compute_cos_sim_pc1_adj_ABO, list_inputs)
