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
from scipy.stats import wilcoxon, norm, kruskal, tukey_hsd, mode, sem
from scipy.io import loadmat
from scipy.spatial.distance import cdist
import seaborn as sns
from copy import deepcopy
from statsmodels.stats.multitest import multipletests
from itertools import combinations, product, permutations
import math

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.multiclass import OneVsOneClassifier
from sklearn.manifold import Isomap
from sklearn.preprocessing import normalize as normr

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

    # dot_xy = np.dot(x, y)
    # norm_x, norm_y = np.linalg.norm(x.astype(np.float32)), np.linalg.norm(y.astype(np.float32))

    # cos_sim = np.dot(normr(np.squeeze(np.squeeze(x).reshape(1, -1))), \
    #                  normr(np.squeeze(np.squeeze(y).reshape(1, -1))))

    return np.dot(x, y) / (np.linalg.norm(x.astype(np.float32)) * np.linalg.norm(y.astype(np.float32)))

# %%
# Function to compute orthogonal & parallel distance
def compute_orth_par_dist(manifold_name1, manifold_name2, rate_12, rate_sorted_mean_coll):
    mean_vector = rate_sorted_mean_coll.loc[:, manifold_name2] - rate_sorted_mean_coll.loc[:, manifold_name1] # mean vector
    mat1 = rate_12.loc[:, manifold_name1].sub(rate_sorted_mean_coll.loc[:, manifold_name1], axis=0) # trial vector
    mat_orth1 = np.array(mat1.apply(lambda x : np.dot(x, mean_vector), axis=0).div(np.dot(mean_vector, mean_vector)))[:, np.newaxis].T * np.array(mean_vector)[:, np.newaxis]
    mat_par1 = mat_orth1 - mat1

    return mat_orth1, mat_par1

# %%
# ABO Neuropixels mean similarity vs. orthogonal variance (RRneuron) (all neurons)

def compute_meansim_orthopar_ABO_RRneuron(slope_ind, target_slope, adjacency_type='geodesic'):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    print(f'target slope = {target_slope:.1f}')

    num_sess = 32
    num_trial_types = 119

    # Iterate over all sessions

    list_mean_sim2_ABO_one_tt = np.zeros((num_sess, num_trial_types, num_trial_types-1)) # permutation of number of stimuli
    list_orthopar2_ABO_one_tt = np.zeros((num_sess, num_trial_types, num_trial_types-1, 2)) # ortho, par
    list_tot_var2_ABO_one_tt = np.zeros((num_sess, num_trial_types, num_trial_types-1, 2))
    for sess_ind in range(num_sess):
        print(f'sess_ind: {sess_ind}')

        rate = list_rate_all[sess_ind].copy()
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

        list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind],
                                        columns=rate_sorted_mean_coll.columns).copy()

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

        rate_sorted_mean_coll[rate_sorted_mean_coll.isna()] = 0  
        rate_sorted_var_coll[rate_sorted_var_coll.isna()] = 0

        # Compute mean and variance of slope-changed data
        rate_mean_RRneuron_coll, rate_var_RRneuron_coll = \
            compute_mean_var_trial_collapse(stm_cnt_dict, rate_RRneuron_dr)
        # FF_RRneuron = rate_var_RRneuron_dr.div(rate_mean_RRneuron_dr)
        # print(FF_RRneuron)
        # print(rate_var_RRneuron_dr)

        # concatenate centroids of all stimuli
        rate_plus_mean = pd.concat([rate_RRneuron_dr, rate_mean_RRneuron_coll], axis=1)

        # Compute geodesic distance matrix
        n_components = 1 # target number of dimensions
        # n_components = rate_RRneuron_dr.shape[0] # target number of dimensions
        n_neighbors = 5 # number of neighbors

        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        
        isomap.fit(rate_plus_mean.T)
        mean_dist_mat_RRneuron = isomap.dist_matrix_[rate_RRneuron_dr.shape[1]:, rate_RRneuron_dr.shape[1]:].copy() # inter-centroid geodesic distance matrix
                        
        # Iterate over all stimuli (~25 min)
        list_mean_sim_one_tt = np.zeros((num_trial_types, num_trial_types-1))
        list_orthopar_one_tt = np.zeros((num_trial_types, num_trial_types-1, 2))
        list_tot_var_one_tt = np.zeros((num_trial_types, num_trial_types-1, 2))
        for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):
            print(f'trial type ind {trial_type_ind}')

            bool_not_tt = rate_sorted_mean_coll.columns != trial_type

            # compute similarity between centroids
            list_mean_sim_one_tt[trial_type_ind] = mean_dist_mat_RRneuron[trial_type_ind, bool_not_tt].copy()

            # compute orthogonal variance against partner stimulus manifold
            for partner_ind, partner_tt in enumerate(rate_sorted_mean_coll.columns[bool_not_tt]):
                rate_pair = rate_RRneuron_dr.loc[:, [trial_type, partner_tt]].copy()
                label_cnt_dict_pair = dict(zip(np.unique(rate_pair.columns, return_counts=True)[0], np.unique(rate_pair.columns, return_counts=True)[1]))
                rate_sorted_mean_coll_pair, rate_sorted_var_coll_pair = compute_mean_var_trial_collapse(label_cnt_dict_pair, rate_pair)

                mat_orth, mat_par = compute_orth_par_dist(trial_type, partner_tt, rate_pair, rate_sorted_mean_coll_pair)
                list_orthopar_one_tt[trial_type_ind, partner_ind] = [np.var(np.linalg.norm(mat_orth.astype(np.float32), axis=0), ddof=1), \
                                                            np.var(np.linalg.norm(mat_par.astype(np.float32), axis=0), ddof=1)]
            
            # compute total variance for normalization
            list_tot_var_one_tt[trial_type_ind] = rate_sorted_var_coll.loc[:, trial_type].mean() 

        list_mean_sim2_ABO_one_tt[sess_ind] = list_mean_sim_one_tt.copy()
        list_orthopar2_ABO_one_tt[sess_ind] = list_orthopar_one_tt.copy()
        list_tot_var2_ABO_one_tt[sess_ind] = list_tot_var_one_tt.copy()

    # Save into a file
    filename = 'D:\\Users\\USER\\Shin Lab\\code\\meansim_orthopar_ABO_allneu_' + adjacency_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_mean_sim2_ABO_one_tt', 'list_orthopar2_ABO_one_tt', 'list_tot_var2_ABO_one_tt'],
                        'list_mean_sim2_ABO_one_tt': list_mean_sim2_ABO_one_tt, 'list_orthopar2_ABO_one_tt': list_orthopar2_ABO_one_tt, 'list_tot_var2_ABO_one_tt': list_tot_var2_ABO_one_tt}, f)

    print("Ended Process", c_proc.name)

# %%
# loading variables

# ABO
with open('resp_matrix_ep_RS_all_32sess_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_all = resp_matrix_ep_RS_all['list_rate_all'].copy()
    list_rate_all_dr = resp_matrix_ep_RS_all['list_rate_all_dr'].copy()
    list_slopes_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_all_an_loglog'].copy()

# %%
# multiprocessing
list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

# ABO
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[slope_ind, target_slope, 'geodesic'] for slope_ind, target_slope in enumerate(list_target_slopes) if slope_ind == 0]
        
        pool.starmap(compute_meansim_orthopar_ABO_RRneuron, list_inputs)
