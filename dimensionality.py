# %%
from pynwb import NWBHDF5IO
from scipy.io import savemat, loadmat
import mat73
import hdf5storage as st
import pickle

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap

from scipy.sparse.linalg import eigsh
from scipy.stats import wilcoxon, mannwhitneyu, sem, linregress
from scipy.spatial.distance import cdist

import seaborn as sns
from copy import deepcopy as dc
from statsmodels.stats.multitest import multipletests

from itertools import combinations, product
import math
import time

# # tell pandas to show all columns when we display a DataFrame
# pd.set_option("display.max_columns", None)

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
# Effective dimensionality
def compute_eff_dim(sess_ind, n_trial_sampling=100):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

    num_trial_types = 119
    rng = np.random.default_rng(sess_ind)

    list_dim_asis = np.zeros(num_trial_types)
    list_dim_RRneuron = np.zeros((len(list_target_slopes), num_trial_types))
    list_dim_global_asis = np.zeros(2)
    list_dim_global_RRneuron = np.zeros((len(list_target_slopes), 2))
    list_dim_sam_asis = np.zeros(n_trial_sampling)
    list_dim_sam_RRneuron = np.zeros((len(list_target_slopes), n_trial_sampling))

    print(f'sess_ind: {sess_ind}')
   
    rate = list_rate_all[sess_ind].copy()
    rate_sorted = rate.sort_index(axis=1)
    stm = rate_sorted.columns.copy()
    
    bool_onscreen = list_rfmet2[sess_ind][:, -1].astype(bool) # deepcopy
    rate_sorted = rate_sorted.loc[bool_onscreen]

    # Multiply by delta t to convert to spike counts
    rate_sorted = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # # trial shuffling
    # rate_shuf = pd.DataFrame(np.zeros_like(rate_sorted), columns=rate_sorted.columns)
    # for neu_ind in range(rate_sorted.shape[0]):
    #     for trial_type_ind, trial_type in enumerate(all_stm_unique):
    #         shuf_inds = rng.permutation(rate_sorted.loc[:, trial_type].shape[1])
    #         rate_shuf.loc[neu_ind, trial_type] = rate_sorted.loc[neu_ind, trial_type].iloc[shuf_inds]    
    # # rate_sorted = rate_shuf.copy()

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    # list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
    #                                 columns=rate_sorted_mean_coll.columns).copy()
    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog_onscreen[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()
    
    # pca
    n_components = rate_sorted.shape[0]
    pca = PCA(n_components=n_components)

    # Compute effective dimensionality for each stimulus
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        cf = np.cov(rate_sorted.loc[:, trial_type])
        list_dim_asis[trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_sorted.shape[0]
    cf_all = np.cov(rate_sorted)
    list_dim_global_asis[0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_sorted.shape[0]
    cf_cen = np.cov(rate_sorted_mean_coll)
    list_dim_global_asis[1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_sorted.shape[0]

    rand_tt_inds = rng.permutation(range(rate.shape[1]))
    rate = rate_sorted.iloc[:, rand_tt_inds].copy()
    for t_sam_ind in range(n_trial_sampling):
        rate_sam = np.full_like(rate_sorted_mean_coll, np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            rate_tt = rate.loc[:, trial_type].copy()
            rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
        cf_sam = np.cov(rate_sam)
        list_dim_sam_asis[t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_sorted.shape[0]

    # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
    rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
    rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

    for slope_ind, target_slope in enumerate(list_target_slopes):
        print(f'target_slope = {target_slope:.1f}')

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
        
        # 1. Use covariance matrix
        
        # Compute effective dimensionality for each stimulus
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            cf = np.cov(rate_RRneuron_dr.loc[:, trial_type])
            list_dim_RRneuron[slope_ind, trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_RRneuron_dr.shape[0]
        cf_all = np.cov(rate_RRneuron_dr)
        list_dim_global_RRneuron[slope_ind, 0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_RRneuron_dr.shape[0]
        cf_cen = np.cov(rate_mean_RRneuron_coll)
        list_dim_global_RRneuron[slope_ind, 1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_RRneuron_dr.shape[0]

        rate_RRneuron_dr = rate_RRneuron_dr.iloc[:, rand_tt_inds].copy()
        for t_sam_ind in range(n_trial_sampling):
            rate_sam = np.full_like(rate_mean_RRneuron_coll, np.nan)
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                rate_tt = rate_RRneuron_dr.loc[:, trial_type].copy()
                rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
            cf_sam = np.cov(rate_sam)
            list_dim_sam_RRneuron[slope_ind, t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_RRneuron_dr.shape[0]

        # # 2. Use eigenvalues
        # for trial_type_ind, trial_type in enumerate(all_stm_unique):
        #     rate_RRneuron_pca = pca.fit_transform(rate_RRneuron_dr.loc[:, trial_type].T).T            
        #     list_dim_RRneuron[slope_ind, trial_type_ind] = (np.sum(pca.explained_variance_))**2 / np.sum(pca.explained_variance_**2)

    # Save into a file
    filename = 'eff_dim_DC_ABO_' + str(sess_ind) + '.pickle'
    filename = 'eff_dim_DC_ABO_onscreen_' + str(sess_ind) + '.pickle'
    # filename = 'eff_dim_DC_ABO_shuf_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_dim_asis', 'list_dim_RRneuron', 'list_dim_global_asis', 'list_dim_global_RRneuron', 'list_dim_sam_asis', 'list_dim_sam_RRneuron'],
                    'list_dim_asis': list_dim_asis, 'list_dim_RRneuron': list_dim_RRneuron, 'list_dim_global_asis': list_dim_global_asis, 'list_dim_global_RRneuron': list_dim_global_RRneuron,
                    'list_dim_sam_asis': list_dim_sam_asis, 'list_dim_sam_RRneuron': list_dim_sam_RRneuron}, f)

    print("Ended Process", c_proc.name)

# %%
# Effective dimensionality (HVA)
def compute_eff_dim_HVA(sess_ind, n_trial_sampling=10):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

    num_trial_types = 119
    rng = np.random.default_rng(sess_ind)

    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']

    list_dim_asis_HVA = {hva: np.zeros(num_trial_types) for hva in list_HVA_names}
    list_dim_RRneuron_HVA = {hva: np.zeros((len(list_target_slopes), num_trial_types)) for hva in list_HVA_names}
    list_dim_global_asis_HVA = {hva: np.zeros(2) for hva in list_HVA_names}
    list_dim_global_RRneuron_HVA = {hva: np.zeros((len(list_target_slopes), 2)) for hva in list_HVA_names}
    list_dim_sam_asis_HVA = {hva: np.zeros(n_trial_sampling) for hva in list_HVA_names}
    list_dim_sam_RRneuron_HVA = {hva: np.zeros((len(list_target_slopes), n_trial_sampling)) for hva in list_HVA_names}

    for area_ind, area in enumerate(list_HVA_names):
        rate = list_rate_all_HVA[area][sess_ind].copy()

        if np.any(rate) == True: # if neurons exist
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

            list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog_HVA[area][sess_ind],
                                          columns=rate_sorted_mean_coll.columns).copy()

            # pca
            n_components = rate_sorted.shape[0]
            pca = PCA(n_components=n_components)

            # Compute effective dimensionality for each stimulus
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                cf = np.cov(rate_sorted.loc[:, trial_type])
                list_dim_asis_HVA[area][trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_sorted.shape[0]
            cf_all = np.cov(rate_sorted)
            list_dim_global_asis_HVA[area][0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_sorted.shape[0]
            cf_cen = np.cov(rate_sorted_mean_coll)
            list_dim_global_asis_HVA[area][1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_sorted.shape[0]

            rand_tt_inds = rng.permutation(range(rate.shape[1]))
            rate = rate_sorted.iloc[:, rand_tt_inds].copy()
            for t_sam_ind in range(n_trial_sampling):
                rate_sam = np.full_like(rate_sorted_mean_coll, np.nan)
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    rate_tt = rate.loc[:, trial_type].copy()
                    rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
                cf_sam = np.cov(rate_sam)
                list_dim_sam_asis_HVA[area][t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_sorted.shape[0]

            # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
            rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
            rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

            for slope_ind, target_slope in enumerate(list_target_slopes):
                start_time = time.time()
                # print(f'target_slope = {target_slope:.1f}')

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
                
                # 1. Use covariance matrix
                
                # Compute effective dimensionality for each stimulus
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    cf = np.cov(rate_RRneuron_dr.loc[:, trial_type])
                    list_dim_RRneuron_HVA[area][slope_ind, trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_RRneuron_dr.shape[0]
                cf_all = np.cov(rate_RRneuron_dr)
                list_dim_global_RRneuron_HVA[area][slope_ind, 0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_sorted.shape[0]
                cf_cen = np.cov(rate_mean_RRneuron_coll)
                list_dim_global_RRneuron_HVA[area][slope_ind, 1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_sorted.shape[0]

                rate_RRneuron_dr = rate_RRneuron_dr.iloc[:, rand_tt_inds].copy()
                for t_sam_ind in range(n_trial_sampling):
                    rate_sam = np.full_like(rate_sorted_mean_coll, np.nan)
                    for trial_type_ind, trial_type in enumerate(all_stm_unique):
                        rate_tt = rate_RRneuron_dr.loc[:, trial_type].copy()
                        rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
                    cf_sam = np.cov(rate_sam)
                    list_dim_sam_RRneuron_HVA[area][slope_ind, t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_sorted.shape[0]

                print(f'sess_ind {sess_ind}, area {area}, target_slope {target_slope:.1f}, duration {(time.time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'eff_dim_DC_ABO_HVA_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_dim_asis_HVA', 'list_dim_RRneuron_HVA', 'list_dim_global_asis_HVA',
                                        'list_dim_global_RRneuron_HVA', 'list_dim_sam_asis_HVA', 'list_dim_sam_RRneuron_HVA'],
                    'list_dim_asis_HVA': list_dim_asis_HVA, 'list_dim_RRneuron_HVA': list_dim_RRneuron_HVA, 'list_dim_global_asis_HVA': list_dim_global_asis_HVA,
                    'list_dim_global_RRneuron_HVA': list_dim_global_RRneuron_HVA, 'list_dim_sam_asis_HVA': list_dim_sam_asis_HVA, 'list_dim_sam_RRneuron_HVA': list_dim_sam_RRneuron_HVA}, f)

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

    sess_inds_qual_all = resp_matrix_ep_RS_all['sess_inds_qual_all'].copy()

# ABO higher visual areas
with open('resp_matrix_ep_HVA_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_HVA_allensdk = pickle.load(f)

    list_rate_all_HVA = dc(resp_matrix_ep_HVA_allensdk['list_rate_all_HVA'])
    list_slopes_all_an_loglog_HVA = dc(resp_matrix_ep_HVA_allensdk['list_slopes_all_an_loglog_HVA'])
    list_empty_sess2 = dc(resp_matrix_ep_HVA_allensdk['list_empty_sess2'])

with open('resp_matrix_ep_naturalmovie_FC_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_naturalmovie = pickle.load(f)
    brain_observatory_sessid = resp_matrix_ep_naturalmovie['brain_observatory_sessid'].copy()
    list_sess_ids = resp_matrix_ep_naturalmovie['list_sess_ids'].copy()

# receptive field metrics for each V1 unit
save_file_name = 'unit_rf_metrics_all.pickle'
with open(save_file_name, 'rb') as f:
    unit_rf_metrics_all = pickle.load(f)
    list_rfmet2 = unit_rf_metrics_all['list_rfmet2'].copy()
    list_slopes_all_an_loglog_onscreen = unit_rf_metrics_all['list_slopes_all_an_loglog_onscreen'].copy()
list_rfmet2 = list_rfmet2[np.isin(list_sess_ids, brain_observatory_sessid)]

# %%
# multiprocessing
num_sess = len(list_rate_all)

# Effective dimensionality
n_trial_sampling = 100
if __name__ == '__main__':

    with mp.Pool() as pool: # set the parameter 'processes' of Pool() if memory error is raised
        list_inputs = [[sess_ind, n_trial_sampling] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_eff_dim, list_inputs)
