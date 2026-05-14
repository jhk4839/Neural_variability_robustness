# %%
# from pynwb import NWBHDF5IO
from scipy.io import savemat, loadmat
import mat73
import h5py
import hdf5storage as st
# from pymatreader import read_mat
import pickle
import os
import warnings
import multiprocessing as mp
import gzip

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, norm, kruskal, tukey_hsd, mode, spearmanr, rankdata, nbinom, poisson
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.special import gammaln

import seaborn as sns
from copy import deepcopy as dc
from statsmodels.discrete.discrete_model import NegativeBinomial
from itertools import combinations, product, permutations, combinations_with_replacement
import math
import random
from time import time
import networkx as nx

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as logit
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, AffinityPropagation, KMeans
from sklearn.decomposition import PCA

from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.stimulus_analysis.receptive_field_mapping import ReceptiveFieldMapping

import cupy as cp
from cupyx.scipy.spatial.distance import cdist as cp_cdist

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

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# %%
# Function for column normalization
def normc(matrix):
    '''For 2D matrix'''

    matrix_normalized = matrix / np.linalg.norm(matrix, axis=0)

    return matrix_normalized

# %%
# replace cdist with this function in windows
def cdist_euc_gpu(a, b):
    a_sq = cp.sum(a**2, axis=1)[:, cp.newaxis]
    b_sq = cp.sum(b**2, axis=1)[cp.newaxis, :]
    ab = cp.dot(a, b.T)
    sq_dists = a_sq + b_sq - 2 * ab
    sq_dists = cp.maximum(sq_dists, 0)
    
    return cp.sqrt(sq_dists)

# %%
# Save receptive field metrics for each neuron
def unit_rfmet(sess_ind):
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    print(f'sess_ind {sess_ind}')
    sess_id = list_sess_ids[sess_ind]
    rate_sorted = list_rate_gb_all[sess_ind].sort_index(axis=1)
    num_neurons = rate_sorted.shape[0]

    # output_dir = 'D:/Users/USER/MATLAB/Allen_Brain_Neuropixels/ecephys_cache_dir/' # put your path for Allen Brain Observatory cache (refer to AllenSDK)
    # resources_dir = Path.cwd().parent / 'resources'
    # DOWNLOAD_LFP = False
    # manifest_path = os.path.join(output_dir, "manifest.json")
    # cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    # sess_data = cache.get_session_data(sess_id)

    rf_mapping = ReceptiveFieldMapping(list_sess_data[sess_ind])
    # list_onscreen = np.zeros(num_neurons, dtype=bool)
    list_rfmet = np.full((num_neurons, 7), np.nan) # azimuth, elevation, width, height, area, pval, on_screen
    for neu_ind in range(num_neurons):
        if neu_ind % 15 == 0:
            print(f'sess_ind {sess_ind}, neu_ind {neu_ind}/{num_neurons-1}')
        list_rfmet[neu_ind] = rf_mapping._get_rf_stats(list_unit_ids_visp[sess_ind][neu_ind]) # visual degrees
        # list_onscreen[neu_ind] = on_screen

    # Save into a file
    filename = 'unit_rf_metrics' + str(sess_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': 'list_rfmet', 'list_rfmet': list_rfmet}, f)

    print("Ended Process", c_proc.name)

# %%
# decoding
def decode_divneu(sess_ind, decoder_type):

    ''' decoder_type is SVM, logit, RF, kNN '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        num_trial_types = 119
        n_splits = 10
        np.random.seed(0)
        
        # print(f'sess_ind: {sess_ind}')

        rate = list_rate_all[sess_ind].copy()
        rate_sorted = rate.sort_index(axis=1)
        stm = rate_sorted.columns.copy()

        # Multiply by delta t to convert to spike counts
        rate_sorted = rate_sorted * 0.25

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        all_stimuli = all_stm_unique.copy()
        train_stimuli = all_stimuli.copy()
        
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

        list_slopes_att = list_slopes_all_att_loglog[sess_ind].copy()
        list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

        # trial order re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        # 1. divide neurons based on intercept per neuron
        neu_div_inds1 = list_slopes_att[1] < np.median(list_slopes_att[1])
        neu_div_inds2 = list_slopes_att[1] > np.median(list_slopes_att[1])
        list_neu_div_inds = [neu_div_inds1, neu_div_inds2]

        # 2. divide neurons based on spontaneous FF per neuron
        list_neu_div_inds = list_neu_div_inds2[sess_ind]

        # decoding cross-validation (as-is)
        stkfold = StratifiedKFold(n_splits=n_splits)

        list_confusion_test = np.full((len(list_neu_div_inds), n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
        list_accuracy = np.full((len(list_neu_div_inds), n_splits), np.nan)
        for div_ind, neu_div_inds in enumerate(list_neu_div_inds):
            label_train = np.repeat(all_stimuli, min_num_trials)
            rate_train = pd.DataFrame(rate_sorted.reshape(rate_sorted.shape[0], -1)[neu_div_inds], columns=label_train)
            
            for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
                start_time = time()
                X_train, X_test = rate_train.T.iloc[train_index].copy(), rate_train.T.iloc[test_index].copy() # train, test data/label
                y_train, y_test = label_train[train_index].copy(), label_train[test_index].copy()

                mean_ = X_train.mean(axis=0)
                X_train = X_train.sub(mean_, axis=1) # train data mean centering
                X_test = X_test.sub(mean_, axis=1)

                if decoder_type == 'SVM':
                    clf = svm.SVC(kernel='linear', max_iter=1000)
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
                list_confusion_test[div_ind, split_ind] = test_confusion_matrix.copy()

                accuracy = accuracy_score(y_test, y_test_pred)
                list_accuracy[div_ind, split_ind] = accuracy
                # print(accuracy)

                print(f'sess_ind: {sess_ind}, div_ind: {div_ind}, split_ind: {split_ind}, duration {(time()-start_time)/60:.2f} min')

        # calculate cross-validation average test confusion matrix/test accuracy
        mean_confusion_test_asis = np.nanmean(list_confusion_test, axis=1)
        # mean_confusion_test_asis = pd.DataFrame(mean_confusion_test_asis, columns=train_stimuli, index=train_stimuli).fillna(0)
        # # print(mean_confusion_test.round(3))
        
        mean_accuracy_asis = np.nanmean(list_accuracy, axis=1)
        # print(round(mean_accuracy, ndigits=3))

    # Save into a file
    filename = decoder_type + '_decoding_divneu_allstim_' + str(sess_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['mean_confusion_test_asis', 'mean_accuracy_asis'],
                     'mean_confusion_test_asis': mean_confusion_test_asis, 'mean_accuracy_asis': mean_accuracy_asis}, f)
                
    print("Ended Process", c_proc.name)

# %%
# RSA across session pairs
def RSA_across_sesspairs_ABO_rf(slope_ind, target_slope, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # number of trial sampling for each stimulus
    num_trials = 50
    num_trial_types = 119
    num_sess = 32

    # print(f'target slope {target_slope:.1f}')

    # Iterate over all sessions

    np.random.seed(0) # match trial order.

    list_RSM_mean_asis = np.full((num_sess, 2, num_trial_types, num_trial_types), np.nan)
    list_RSM_mean_RRneuron = np.full((num_sess, 2, num_trial_types, num_trial_types), np.nan)
    list_rate_RRneuron_dr = np.empty((num_sess, 2), dtype=object)

    # predetermine random trial order
    list_rand_trial_inds = np.empty((num_sess, num_trial_types), dtype=object)
    for ind in range(num_sess):
        rate = list_rate_all[ind].copy()
        stm = rate.columns.copy()

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)        
        for trial_type_ind in range(num_trial_types):
            list_rand_trial_inds[ind, trial_type_ind] = np.random.choice(range(min_num_trials), min_num_trials, replace=False)

    for ind in range(num_sess):
        # if np.min(list_num_neurons_lr[ind, :-1]) >= 10: # if both halves have >=10 neurons
        if np.min(list_num_neurons_ud[ind, :-1]) >= 10: # if both halves have >=10 neurons
            
            rate_all = list_rate_all[ind].copy()
            # rate_sorted = rate_all.sort_index(axis=1)
            stm = rate_all.columns.copy()

            # exclude neurons out of the screen, and divide neurons based on rf position
            azi = list_rfmet2[ind][:, 0].copy()
            elev = list_rfmet2[ind][:, 1].copy()
            onscreen = list_rfmet2[ind][:, -1].copy()
            rf_cen_within = ((azi >= 10) & (azi <= 90)) & ((elev >= -30) & (elev <= 50))
            onscreen_real = rf_cen_within & (onscreen == 1)
            azi_left, azi_right = onscreen_real & (azi < 50), onscreen_real & (azi > 50)
            elev_up, elev_down = onscreen_real & (elev > 10), onscreen_real & (elev < 10)
            list_bool_azi = dc([azi_left, azi_right])
            list_bool_elev = dc([elev_up, elev_down])
            
            for div_ind in range(list_RSM_mean_asis.shape[1]):
                print(f'target slope {target_slope:.1f}, ind: {ind}, div_ind {div_ind}')
                
                # rate = rate_all.loc[list_bool_azi[div_ind]].copy()
                # num_neu_now = np.sum(list_bool_azi[div_ind])
                # num_neu_opp = np.sum(list_bool_azi[1-div_ind])
                rate = rate_all.loc[list_bool_elev[div_ind]].copy()
                num_neu_now = np.sum(list_bool_elev[div_ind])
                num_neu_opp = np.sum(list_bool_elev[1-div_ind])

                # match number of neurons in both halves
                if num_neu_now > num_neu_opp:
                    rate = rate.iloc[np.random.choice(range(num_neu_now), num_neu_opp, replace=False)]
                
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
                
                list_slopes_dr = list_slopes_all_an_loglog_onscreen_real[ind].copy()

                # # trial shuffling
                # rate_shuf = np.zeros_like(rate_sorted)
                # for neu_ind in range(rate_sorted.shape[0]):
                #     shuf_inds = np.random.permutation(rate_sorted.shape[2])
                #     rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy()
                # rate_sorted = rate_shuf.copy()

                # trial order re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, list_rand_trial_inds[ind, trial_type_ind]]

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

                        # RSM 제작
                        RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2))

                        # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos))
                        list_RSM[sampling_ind] = RSM.copy()
                        
                        count += 1
                        if count % (n_sampling//2) == 0:
                            print(f'target slope {target_slope:.1f}, ind: {ind}, div_ind {div_ind}, count: {count}')

                    RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
                    list_RSM_mean_asis[ind, div_ind] = RSM_mean.copy()

                # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

                # RRneuron

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

                list_rate_RRneuron_dr[ind, div_ind] = rate_RRneuron_dr.copy()

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
                    if count % (n_sampling//2) == 0:
                        print(f'target slope {target_slope:.1f}, ind: {ind}, div_ind {div_ind}, count: {count}')

                RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
                list_RSM_mean_RRneuron[ind, div_ind] = RSM_mean.copy()

    # Spearman correlation across session pairs (only between different halves)
    sess_pairs = list(combinations(range(num_sess), 2))
    list_corr_sesspair_asis = np.zeros((len(sess_pairs), 2, 3)) 
    list_corr_sesspair = np.zeros((len(sess_pairs), 2, 3)) 
    for pair_ind, pair in enumerate(sess_pairs):
        for div_ind in range(list_corr_sesspair_asis.shape[1]):
            if slope_ind == 0:
                RSM_mean_neu1 = list_RSM_mean_asis[pair[0], div_ind].copy()
                RSM_mean_neu2 = list_RSM_mean_asis[pair[1], 1-div_ind].copy()

                list_corr_sesspair_asis[pair_ind, div_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_sesspair_asis[pair_ind, div_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_sesspair_asis[pair_ind, div_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
            
            RSM_mean_neu1 = list_RSM_mean_RRneuron[pair[0], div_ind].copy()
            RSM_mean_neu2 = list_RSM_mean_RRneuron[pair[1], 1-div_ind].copy()

            list_corr_sesspair[pair_ind, div_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_sesspair[pair_ind, div_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_sesspair[pair_ind, div_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_sesspair_ABO_allneu_rf_lr_' + similarity_type + str(slope_ind) + '.pickle.gz'
    filename = 'RSM_corr_sesspair_ABO_allneu_rf_ud_' + similarity_type + str(slope_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis', 'list_corr_sesspair_asis', 'list_rate_RRneuron_dr', 'list_RSM_mean_RRneuron', 'list_corr_sesspair'], \
                     'list_RSM_mean_asis': list_RSM_mean_asis, 'list_corr_sesspair_asis': list_corr_sesspair_asis,
                     'list_rate_RRneuron_dr': list_rate_RRneuron_dr, 'list_RSM_mean_RRneuron': list_RSM_mean_RRneuron, 'list_corr_sesspair': list_corr_sesspair}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA across session pairs
def RSA_across_sesspairs_ABO_divneu(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_trial_types = 119

    rng = np.random.default_rng(sess_ind) # match trial order.

    print(f'sess_ind {sess_ind}')

    rate = list_rate_all[sess_ind].copy()
    # rate_sorted = rate.sort_index(axis=1)
    stm = rate.columns.copy()
    num_neurons = rate.shape[0]

    # Multiply by delta t to convert to spike counts
    rate = rate * 0.25
    
    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
    
    # convert to 3D response matrix
    min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

    list_rate_tt = [None] * num_trial_types
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

    rate = np.stack(list_rate_tt, axis=2)
    rate_sorted_all = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials

    rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted_all, axis=2), np.var(rate_sorted_all, axis=2, ddof=1)
    rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
        np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
    
    list_slopes_att = list_slopes_all_att_loglog[sess_ind].copy()
    list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

    # # 1. divide neurons based on intercept per neuron
    # neu_div_inds1 = list_slopes_att[1] < np.median(list_slopes_att[1])
    # neu_div_inds2 = list_slopes_att[1] > np.median(list_slopes_att[1])

    # 2. divide neurons based on spontaneous FF per neuron
    list_neu_div_inds = list_neu_div_inds2[sess_ind]

    # trial order re-randomization
    for trial_type_ind in range(num_trial_types):
        rate_sorted_all[:, trial_type_ind, :] = rate_sorted_all[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

    list_RSM_mean_asis = np.zeros((len(list_neu_div_inds), num_trial_types, num_trial_types))
    for div_ind, neu_div_inds in enumerate(list_neu_div_inds):
        rate_sorted = rate_sorted_all[neu_div_inds].copy()

        # repeat calculating similarity matrices

        # n_neurons x n_stimuli 2D matrix sampling
        tt_pairs = list(combinations(range(min_num_trials), 2))
        n_sampling = np.min([len(tt_pairs), 10000])
        # n_sampling = len(tt_pairs)

        list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
        
        count = 0
        for sampling_ind in range(n_sampling):
            rate_sampled_trials1 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][0]]).copy()
            rate_sampled_trials2 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][1]]).copy()

            # RSM 제작
            RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2))

            # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos))
            list_RSM[sampling_ind] = RSM.copy()
            
            count += 1
            if count % (n_sampling//2) == 0:
                print(f'count: {count}')

        RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
        list_RSM_mean_asis[div_ind] = RSM_mean.copy()

    # Save into a file
    filename = 'RSM_divneu_allneu_' + similarity_type + str(sess_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': 'list_RSM_mean_asis', 'list_RSM_mean_asis': list_RSM_mean_asis}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA within sessions
def RSA_withinsess_ABO_rf(slope_ind, target_slope, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # number of trial sampling for each stimulus
    num_trials = 50
    num_trial_types = 119
    num_sess = 32

    # print(f'target slope {target_slope:.1f}')

    # Iterate over all sessions
    np.random.seed(0) # match neuron partitioning
    # random.seed(0)

    list_corr_withinsess_asis2 = np.full((num_sess, 3), np.nan)
    list_corr_withinsess2 = np.full((num_sess, 3), np.nan)

    # predetermine random trial order
    list_shuf_inds = np.empty(num_sess, dtype=object)
    list_rand_trial_inds = np.empty((num_sess, num_trial_types), dtype=object)
    for sess_ind in range(num_sess):
        rate = list_rate_all[sess_ind].copy()
        stm = rate.columns.copy()

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)        
        
        shuf_inds_temp = []
        for neu_ind in range(rate.shape[0]):
            shuf_inds_temp.append(np.random.permutation(min_num_trials))
        list_shuf_inds[sess_ind] = dc(shuf_inds_temp)

        for trial_type_ind in range(num_trial_types):
            list_rand_trial_inds[sess_ind, trial_type_ind] = np.random.choice(range(min_num_trials), min_num_trials, replace=False)

    for sess_ind in range(num_sess):
        # if np.min(list_num_neurons_lr[sess_ind, :-1]) >= 10: # if both halves have >=10 neurons
        if np.min(list_num_neurons_ud[sess_ind, :-1]) >= 10: # if both halves have >=10 neurons
            print(f'target slope {target_slope:.1f}, sess_ind: {sess_ind}')

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
                        
            list_slopes_dr = list_slopes_all_an_loglog_onscreen_real[sess_ind].copy()

            # trial shuffling
            rate_shuf = np.zeros_like(rate_sorted)
            for neu_ind in range(rate_sorted.shape[0]):
                shuf_inds = list_shuf_inds[sess_ind][neu_ind].copy()
                rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy()
            # rate_sorted = rate_shuf.copy()

            # trial order re-randomization
            for trial_type_ind in range(num_trial_types):
                rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, list_rand_trial_inds[sess_ind, trial_type_ind]]

            # exclude neurons out of the screen, and divide neurons based on rf position
            azi = list_rfmet2[sess_ind][:, 0].copy()
            elev = list_rfmet2[sess_ind][:, 1].copy()
            onscreen = list_rfmet2[sess_ind][:, -1].copy()
            rf_cen_within = ((azi >= 10) & (azi <= 90)) & ((elev >= -30) & (elev <= 50))
            onscreen_real = rf_cen_within & (onscreen == 1)
            azi_left, azi_right = onscreen_real & (azi < 50), onscreen_real & (azi > 50)
            elev_up, elev_down = onscreen_real & (elev > 10), onscreen_real & (elev < 10)
            list_bool_azi = dc([azi_left, azi_right])
            list_bool_elev = dc([elev_up, elev_down])            
            
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

            # divide neurons
            # neu_div_inds1 = np.where(azi_left)[0]
            # neu_div_inds2 = np.where(azi_right)[0]
            # num_neu1 = np.sum(azi_left)
            # num_neu2 = np.sum(azi_right)
            neu_div_inds1 = np.where(elev_up)[0]
            neu_div_inds2 = np.where(elev_down)[0]
            num_neu1 = np.sum(elev_up)
            num_neu2 = np.sum(elev_down)
                                    
            # match number of neurons in both halves
            if num_neu1 > num_neu2:
                neu_div_inds1 = np.random.choice(neu_div_inds1, num_neu2, replace=False)
            elif num_neu1 < num_neu2:
                neu_div_inds2 = np.random.choice(neu_div_inds2, num_neu1, replace=False)

            # as-is
            if slope_ind == 0:

                # repeat calculating similarity matrices
                
                # n_neurons x n_stimuli 2D matrix sampling
                
                tt_pairs = list(combinations(range(min_num_trials), 2))
                # random.shuffle(tt_pairs)
                n_sampling = np.min([len(tt_pairs), 10000])
                # n_sampling = len(tt_pairs)

                list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32) # left
                list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32) # right

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

                RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
                RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                list_corr_withinsess_asis2[sess_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_withinsess_asis2[sess_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_withinsess_asis2[sess_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
        
            # RRneuron

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
                if count % (n_sampling//2) == 0:
                    print(f'count: {count}')

            RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
            RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
            list_corr_withinsess2[sess_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_withinsess2[sess_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_withinsess2[sess_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_withinsess_ABO_rf_lr_' + similarity_type + str(slope_ind) + '.pickle.gz'
    filename = 'RSM_corr_withinsess_ABO_rf_ud_' + similarity_type + str(slope_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis2', 'list_corr_withinsess2'], \
                    'list_corr_withinsess_asis2': list_corr_withinsess_asis2, 'list_corr_withinsess2': list_corr_withinsess2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# RSA within sessions
def RSA_withinsess_ABO_divneu(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_trial_types = 119
    n_neu_sampling = 10

    # Iterate over all sessions
    rng = np.random.default_rng(sess_ind) # match neuron partitioning
    # random.seed(0)

    # print(f'sess_ind {sess_ind}')

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
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

    rate = np.stack(list_rate_tt, axis=2)
    rate_sorted_all = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials

    rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted_all, axis=2), np.var(rate_sorted_all, axis=2, ddof=1)
    rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
        np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
    
    list_slopes_att = list_slopes_all_att_loglog[sess_ind].copy()
    list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

    # # 1. divide neurons based on intercept per neuron
    # neu_div_inds_cri1 = list_slopes_att[1] < np.median(list_slopes_att[1])
    # neu_div_inds_cri2 = list_slopes_att[1] > np.median(list_slopes_att[1])

    # 2. divide neurons based on spontaneous FF per neuron
    list_neu_div_inds = list_neu_div_inds2[sess_ind]

    # trial order re-randomization
    for trial_type_ind in range(num_trial_types):
        rate_sorted_all[:, trial_type_ind, :] = rate_sorted_all[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

    # Iterate over neuron partitionings
    list_corr_withinsess_asis = np.full((len(list_neu_div_inds), n_neu_sampling, 3), np.nan)
    list_RSM_neu1_all = np.full((len(list_neu_div_inds), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu2_all = np.full((len(list_neu_div_inds), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    for div_ind, neu_div_inds in enumerate(list_neu_div_inds):
        print(f'sess_ind {sess_ind}, div_ind = {div_ind}')
        rate_sorted = rate_sorted_all[neu_div_inds].copy()
        
        for neu_sample_ind in range(n_neu_sampling):
            # Partition neurons
            neu_inds_permuted = rng.permutation(range(rate_sorted.shape[0]))
            neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 partitioning
            neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
            if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
                neu_div_inds2 = neu_div_inds2[:-1].copy()
                
            # as-is

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

            RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
            RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
            list_RSM_neu1_all[div_ind, neu_sample_ind], list_RSM_neu2_all[div_ind, neu_sample_ind] = RSM_mean_neu1.copy(), RSM_mean_neu2.copy()

            list_corr_withinsess_asis[div_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_withinsess_asis[div_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_withinsess_asis[div_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_withinsess_divneu_' + similarity_type + str(sess_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:      
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis', 'list_RSM_neu1_all', 'list_RSM_neu2_all'], \
                    'list_corr_withinsess_asis': list_corr_withinsess_asis, 'list_RSM_neu1_all': list_RSM_neu1_all, 'list_RSM_neu2_all': list_RSM_neu2_all}, f)
        
    print("Ended Process", c_proc.name)

# %%
# RSA within sessions
def RSA_withinsess_ABO_vis(slope_ind, target_slope, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # number of trial sampling for each stimulus
    num_trials = 50
    num_trial_types = 119
    num_sess = 32
    n_neu_sampling = 10
    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']
    list_vis_names = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']
    vis_pairs = list(combinations(list_vis_names, 2))

    list_rate_all_vis = dc(list_rate_all_HVA)
    list_rate_all_vis['VISp'] = list_rate_all
    list_slopes_all_an_loglog_vis = dc(list_slopes_all_an_loglog_HVA)
    list_slopes_all_an_loglog_vis['VISp'] = list_slopes_all_an_loglog
    list_num_neurons_vis = dc(list_num_neurons_HVA)
    list_num_neurons_vis['VISp'] = list_num_neurons_visp

    # predetermine random order
    
    # V1
    np.random.seed(0)
    # list_shuf_inds = np.empty(num_sess, dtype=object)
    list_rand_trial_inds = np.empty((num_sess, num_trial_types), dtype=object)
    # list_neu_div_inds = np.empty(num_sess, dtype=object)
    for sess_ind in range(num_sess):
        rate = list_rate_all[sess_ind].copy()
        stm = rate.columns.copy()

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)        
        
        # shuf_inds_temp = []
        # for neu_ind in range(rate.shape[0]):
        #     shuf_inds_temp.append(np.random.permutation(min_num_trials))
        # list_shuf_inds[sess_ind] = dc(shuf_inds_temp)

        for trial_type_ind in range(num_trial_types):
            list_rand_trial_inds[sess_ind, trial_type_ind] = np.random.choice(range(min_num_trials), min_num_trials, replace=False)

        # list_neu_div_inds_temp = np.empty(n_neu_sampling, dtype=object)
        # for neu_sample_ind in range(n_neu_sampling):
        #     # Partition neurons
        #     neu_inds_permuted = np.random.permutation(range(rate.shape[0]))
        #     neu_div_inds1 = neu_inds_permuted[:int(rate.shape[0]/2)].copy() # 5:5 partitioning
        #     neu_div_inds2 = neu_inds_permuted[int(rate.shape[0]/2):].copy()
        #     if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
        #         neu_div_inds2 = neu_div_inds2[:-1].copy()
        #     list_neu_div_inds_temp[neu_sample_ind] = [neu_div_inds1, neu_div_inds2]
        # list_neu_div_inds[sess_ind] = list_neu_div_inds_temp

    list_corr_withinsess_asis3 = np.full((len(vis_pairs), num_sess, n_neu_sampling, 3), np.nan)
    list_corr_withinsess3 = np.full((len(vis_pairs), num_sess, n_neu_sampling, 3), np.nan)
    for area_pair_ind, area_pair in enumerate(vis_pairs):
        for sess_ind in range(num_sess):
            if list_num_neurons_vis[area_pair[0]][sess_ind] > 0 and list_num_neurons_vis[area_pair[1]][sess_ind] > 0:
                # print(f'target slope {target_slope:.1f}, area_pair_ind {area_pair_ind}/{len(vis_pairs)-1}, sess_ind: {sess_ind}')
                
                start_time = time()
                rate_pair, rate_pair_RRneuron = [], []
                for i, area in enumerate(area_pair):
                    rate = list_rate_all_vis[area][sess_ind]
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
                                
                    list_slopes_dr = list_slopes_all_an_loglog_vis[area_pair[i]][sess_ind].copy()

                    # trial order re-randomization
                    for trial_type_ind in range(num_trial_types):
                        rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, list_rand_trial_inds[sess_ind, trial_type_ind]]
                    
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

                    rate_pair.append(rate_sorted)
                    rate_pair_RRneuron.append(rate_RRneuron_dr)

                # Iterate over neuron partitionings
                for neu_sample_ind in range(n_neu_sampling):
                    # print(f'neu_sample_ind = {neu_sample_ind}')
                    
                    # Partition neurons
                    list_neu_div_inds = []
                    num_neurons_half = np.min([rate_pair[i].shape[0] for i in range(2)]) // 2 # to match number of neurons between the two areas
                    for i in range(len(rate_pair)):
                        num_neurons_temp = rate_pair[i].shape[0]
                        neu_inds_permuted = np.random.permutation(range(num_neurons_temp))
                        neu_div_inds = neu_inds_permuted[:num_neurons_half]
                        list_neu_div_inds.append(neu_div_inds)
                    neu_div_inds1, neu_div_inds2 = list_neu_div_inds

                    # as-is
                    if slope_ind == 0:
                        # n_neurons x n_stimuli 2D matrix sampling
                        
                        tt_pairs = list(combinations(range(min_num_trials), 2))
                        # random.shuffle(tt_pairs)
                        n_sampling = np.min([len(tt_pairs), 10000])
                        # n_sampling = len(tt_pairs)

                        list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                        list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                        count = 0
                        for sampling_ind in range(n_sampling):
                            rate_sampled_trials1_1 = np.squeeze(rate_pair[0][neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy()
                            rate_sampled_trials1_2 = np.squeeze(rate_pair[0][neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                            rate_sampled_trials2_1 = np.squeeze(rate_pair[1][neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy()
                            rate_sampled_trials2_2 = np.squeeze(rate_pair[1][neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                            RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2))
                            RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2))

                            list_RSM_neu1[sampling_ind] = RSM1.copy()
                            list_RSM_neu2[sampling_ind] = RSM2.copy()

                        RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
                        RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                        list_corr_withinsess_asis3[area_pair_ind, sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                        bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                        list_corr_withinsess_asis3[area_pair_ind, sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                        list_corr_withinsess_asis3[area_pair_ind, sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
                
                    # RRneuron

                    # n_neurons x n_stimuli 2D matrix sampling
                    tt_pairs = list(combinations(range(min_num_trials), 2))
                    # random.shuffle(tt_pairs)
                    n_sampling = np.min([len(tt_pairs), 10000])
                    # n_sampling = len(tt_pairs)

                    list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                    list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                    count = 0
                    for sampling_ind in range(n_sampling):
                        rate_sampled_trials1_1 = np.squeeze(rate_pair_RRneuron[0][neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy()
                        rate_sampled_trials1_2 = np.squeeze(rate_pair_RRneuron[0][neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                        rate_sampled_trials2_1 = np.squeeze(rate_pair_RRneuron[1][neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy()
                        rate_sampled_trials2_2 = np.squeeze(rate_pair_RRneuron[1][neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                        RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2))
                        RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2))

                        # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos))
                        list_RSM_neu1[sampling_ind] = RSM1.copy()
                        list_RSM_neu2[sampling_ind] = RSM2.copy()

                        count += 1
                        # if count % (n_sampling//2) == 0:
                        #     print(f'count: {count}')

                    RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
                    RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                    list_corr_withinsess3[area_pair_ind, sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                    bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                    list_corr_withinsess3[area_pair_ind, sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                    list_corr_withinsess3[area_pair_ind, sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

                print(f'target slope {target_slope:.1f}, area_pair_ind {area_pair_ind}/{len(vis_pairs)-1}, sess_ind: {sess_ind}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'RSM_corr_withinsess_ABO_vis_' + similarity_type + str(slope_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis3', 'list_corr_withinsess3'], \
                    'list_corr_withinsess_asis3': list_corr_withinsess_asis3, 'list_corr_withinsess3': list_corr_withinsess3}, f)
        
    print("Ended Process", c_proc.name)

# %%
# Effective dimensionality
def compute_eff_dim(sess_ind, n_trial_sampling=100):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    rng = np.random.default_rng(sess_ind)

    print(f'sess_ind {sess_ind}')
    
    rate = list_rate_all[sess_ind].copy()
    rate_sorted = rate.sort_index(axis=1)
    stm = rate_sorted.columns.copy()

    # Multiply by delta t to convert to spike counts
    rate_sorted_all = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean_all, rate_sorted_var_all = compute_mean_var_trial(stm_cnt_dict, rate_sorted_all)
    rate_sorted_mean_coll_all, rate_sorted_var_coll_all = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted_all)

    list_slopes_att = pd.DataFrame(list_slopes_all_att_loglog[sess_ind], columns=rate_sorted_mean_coll_all.index).copy()
    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll_all.columns).copy()

    # # 1. divide neurons based on intercept per neuron
    # neu_div_inds1 = list_slopes_att.loc[1] < np.median(list_slopes_att.loc[1])
    # neu_div_inds2 = list_slopes_att.loc[1] > np.median(list_slopes_att.loc[1])
    # list_neu_div_inds = [neu_div_inds1, neu_div_inds2]

    # 2. divide neurons based on spontaneous FF per neuron
    list_neu_div_inds = list_neu_div_inds2[sess_ind]

    list_dim_asis = np.zeros((len(list_neu_div_inds), num_trial_types))
    list_dim_global_asis = np.zeros((len(list_neu_div_inds), 2))
    list_dim_sam_asis = np.zeros((len(list_neu_div_inds), n_trial_sampling))
    for div_ind, neu_div_inds in enumerate(list_neu_div_inds):
        print(f'sess_ind {sess_ind}, div_ind = {div_ind}')
        rate_sorted = rate_sorted_all.loc[neu_div_inds].copy()
        rate_sorted_mean_coll = rate_sorted_mean_coll_all.loc[neu_div_inds].copy()

        # pca
        n_components = rate_sorted.shape[0]
        pca = PCA(n_components=n_components)

        # Compute effective dimensionality for each stimulus
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            cf = np.cov(rate_sorted.loc[:, trial_type])
            list_dim_asis[div_ind, trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_sorted.shape[0]
        cf_all = np.cov(rate_sorted)
        list_dim_global_asis[div_ind, 0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_sorted.shape[0]
        cf_cen = np.cov(rate_sorted_mean_coll)
        list_dim_global_asis[div_ind, 1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_sorted.shape[0]

        rand_tt_inds = rng.permutation(range(rate_sorted.shape[1]))
        rate = rate_sorted.iloc[:, rand_tt_inds].copy()
        for t_sam_ind in range(n_trial_sampling):
            rate_sam = np.full_like(rate_sorted_mean_coll, np.nan)
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                rate_tt = rate.loc[:, trial_type].copy()
                rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
            cf_sam = np.cov(rate_sam)
            list_dim_sam_asis[div_ind, t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_sorted.shape[0]

    # Save into a file
    filename = 'eff_dim_DC_divneu_' + str(sess_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_dim_asis', 'list_dim_global_asis', 'list_dim_sam_asis'],
                    'list_dim_asis': list_dim_asis, 'list_dim_global_asis': list_dim_global_asis, 'list_dim_sam_asis': list_dim_sam_asis}, f)

    print("Ended Process", c_proc.name)

# %%
# compare slopes per stimulus for different neuronal groups
def fit_slopes_divneu(sess_ind):
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    print(f'session index: {sess_ind}')

    rng = np.random.default_rng(sess_ind)
    
    rate = list_rate_all[sess_ind].copy()
    rate_sorted = rate.sort_index(axis=1)
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

    list_slopes_att = pd.DataFrame(list_slopes_all_att_loglog[sess_ind], columns=rate_sorted_mean_coll.index).copy()
    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()

    # # 1. divide neurons based on slope/intercept per neuron

    # # boundaries = np.percentile(list_slopes_att.loc[1], [100/3, 200/3])
    # # neu_bin_inds = np.digitize(list_slopes_att.loc[1], boundaries)
    # # neu_bin_unq, num_neurons_bins = np.unique(neu_bin_inds, return_counts=True)
    # # neu_div_inds1 = np.where(neu_bin_inds == 0)[0]
    # # neu_div_inds2 = np.where(neu_bin_inds == 1)[0]
    # # neu_div_inds3 = np.where(neu_bin_inds == 2)[0]

    # # min_num_neurons = np.min(num_neurons_bins)
    # # list_neu_div_inds_temp = dc([neu_div_inds1, neu_div_inds2, neu_div_inds3])
    # # list_neu_div_inds = np.empty(len(neu_bin_unq), dtype=object)
    # # for div_ind, neu_div_inds in enumerate(list_neu_div_inds_temp):
    # #     list_neu_div_inds[div_ind] = neu_div_inds[:min_num_neurons].copy()
    # # # print(len(neu_div_inds1), len(neu_div_inds2), len(neu_div_inds3))

    # boundaries = np.percentile(list_slopes_att.values[1], 50)
    # neu_div_inds1 = np.where(list_slopes_att.values[1] < boundaries)[0]
    # neu_div_inds3 = np.where(list_slopes_att.values[1] > boundaries)[0]
    # neu_div_inds2 = np.sort(rng.choice(range(num_neurons), len(neu_div_inds1), replace=False))
    # list_neu_div_inds = [neu_div_inds1, neu_div_inds2, neu_div_inds3]

    # 2. divide neurons based on spontaneous FF per neuron

    sess_ind_mapped = sess_inds_mapBO[sess_ind]
    rate_sorted_spt = list_rate_spt_BO_all[sess_ind_mapped]
    cri_spt = np.var(rate_sorted_spt, axis=1, ddof=1) / np.mean(rate_sorted_spt, axis=1)
    # print(np.sum(np.isinf(cri_spt))) # confirmed no neurons with mean zero in all sessions

    boundaries = np.percentile(cri_spt, 50)
    # neu_bin_inds = np.digitize(cri_spt, [boundaries])
    # neu_div_inds1 = np.where(neu_bin_inds == 0)[0]
    # neu_div_inds2 = np.where(neu_bin_inds == 1)[0]
    neu_div_inds1 = np.where(cri_spt < boundaries)[0]
    neu_div_inds3 = np.where(cri_spt > boundaries)[0]
    neu_div_inds2 = np.sort(rng.choice(range(num_neurons), len(neu_div_inds1), replace=False))
    list_neu_div_inds = dc([neu_div_inds1, neu_div_inds2, neu_div_inds3])
    # print(len(neu_div_inds1), len(neu_div_inds2))

    # fit slope per stimulus for each group
    slopes = np.zeros((len(list_neu_div_inds), 2, rate_sorted_mean_coll.shape[1]))
    for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):
        for div_ind, neu_div_inds in enumerate(list_neu_div_inds):
            mean_temp = rate_sorted_mean.loc[neu_div_inds].copy()
            var_temp = rate_sorted_var.loc[neu_div_inds].copy()

            # loglog scale
            bool_mean_notzero = rate_sorted_mean_coll.loc[neu_div_inds, trial_type] > 0
            popt = np.polyfit(np.log10(mean_temp.loc[bool_mean_notzero, trial_type].values).flatten().astype(np.float32), \
                                np.log10(var_temp.loc[bool_mean_notzero, trial_type].values).flatten().astype(np.float32), 1)
            
            slopes[div_ind, :, trial_type_ind] = popt.copy()
    
    # Save into a file
    filename = 'slopes_divneu_' + str(sess_ind) + '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['slopes', 'list_neu_div_inds'],
                    'slopes': slopes, 'list_neu_div_inds': list_neu_div_inds}, f)

    print("Ended Process", c_proc.name)

# %%
# randomly sample trials and compute overlap consistency
def compute_overlap_stimpairs_consis(sess_ind, n_trial_sampling=10):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    num_sampling = 10 # Number of random neuron partitioning
    list_num_trials = [10, 25]

    print(f'sess_ind: {sess_ind}')
    
    np.random.seed(0)

    rate = list_rate_all[sess_ind].copy()
    stm = rate.columns.copy()
    num_neurons = rate.shape[0]

    # Multiply by delta t to convert to spike counts
    rate = rate * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # convert to 3D response matrix
    min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

    list_rate_tt = [None] * num_trial_types
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

    rate = np.stack(list_rate_tt, axis=2)
    rate_sorted_all = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x min_num_trials

    # Compute mean & variance for each stimulus
    rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted_all, axis=2), np.var(rate_sorted_all, axis=2, ddof=1)
    rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
        np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)

    list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

    # neuron partitioning
    list_neu_div_inds2 = np.zeros((num_sampling, 2, rate_sorted_all.shape[0]//2), dtype=int)
    for sampling_ind in range(num_sampling):
        # Partition neurons
        neu_inds_permuted = np.random.permutation(range(rate_sorted_all.shape[0]))
        neu_div_inds1 = neu_inds_permuted[:int(rate_sorted_all.shape[0]/2)].copy() # 5:5 partitioning
        neu_div_inds2 = neu_inds_permuted[int(rate_sorted_all.shape[0]/2):].copy()
        if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
            neu_div_inds2 = neu_div_inds2[:-1].copy()
        list_neu_div_inds2[sampling_ind] = dc([neu_div_inds1, neu_div_inds2])

    # trial order re-randomization
    for trial_type_ind in range(num_trial_types):
        rate_sorted_all[:, trial_type_ind, :] = rate_sorted_all[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

    # trial sampling
    list_rand_trial_inds2 = np.empty(len(list_num_trials), dtype=object)
    for t_ind, num_trials in enumerate(list_num_trials):
        list_rand_trial_inds = np.zeros((n_trial_sampling, num_trials), dtype=int)
        for t_sam_ind in range(n_trial_sampling):
            list_rand_trial_inds[t_sam_ind] = np.random.choice(range(min_num_trials), num_trials, replace=False)
        list_rand_trial_inds2[t_ind] = list_rand_trial_inds.copy()

    list_overlap_asis2 = np.full((num_sampling, 2, len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    list_overlap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    # list_gap_asis2 = np.full((num_sampling, 2, len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    # list_gap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    for sampling_ind in range(num_sampling):
        if sampling_ind == 0:
            # print(f'sess_ind = {sess_ind}, sampling ind = {sampling_ind}')

            # Partition neurons
            list_neu_div_inds = list_neu_div_inds2[sampling_ind].copy()
            neu_div_inds1, neu_div_inds2 = list_neu_div_inds.copy()

            for t_ind, num_trials in enumerate(list_num_trials):
                for t_sam_ind in range(n_trial_sampling):
                    rate_sorted = rate_sorted_all[:, :, list_rand_trial_inds2[t_ind][t_sam_ind]].copy()
                    rate_sorted = pd.DataFrame(rate_sorted.reshape(num_neurons, -1), columns=np.repeat(all_stm_unique, num_trials))

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
                            list_overlap_asis2[sampling_ind, div_ind, t_ind, t_sam_ind, trial_type_ind] = overlap.copy()

                            # gap = np.min(nbr_dist, axis=1).groupby(level=0).min()
                            # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                            # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                            # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                            # # print(gap.shape)
                            # # list_gap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[div_ind, all_stm_unique != trial_type, 1]
                            # list_gap_asis[div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]

            # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
            rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
            rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
            
            for slope_ind, target_slope in enumerate(list_target_slopes):
                if slope_ind == 0:
                    start_time = time()
                    # print(f'sampling ind = {sampling_ind}, target slope = {target_slope:.1f}')

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
                    rate_sorted_resid_dr = rate_sorted_all - rate_sorted_mean
                    # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                    #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                    rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                        * np.sqrt(var_rs_noisy)
                    # print(rate_resid_RRneuron_dr)
                    rate_RRneuron_dr_all = rate_sorted_mean + rate_resid_RRneuron_dr
                    rate_RRneuron_dr_all[np.isnan(rate_RRneuron_dr_all)] = 0 # convert NaN to 0!

                    for t_ind, num_trials in enumerate(list_num_trials):
                        for t_sam_ind in range(n_trial_sampling):
                            rate_RRneuron_dr = rate_RRneuron_dr_all[:, :, list_rand_trial_inds2[t_ind][t_sam_ind]].copy()
                            rate_RRneuron_dr = pd.DataFrame(rate_RRneuron_dr.reshape(num_neurons, -1), columns=np.repeat(all_stm_unique, num_trials))                    
                            
                            # Determine criteria using internal pairwise distance for each stimulus

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
                                    list_overlap_RRneuron3[sampling_ind, slope_ind, div_ind, t_ind, t_sam_ind, trial_type_ind] = overlap.copy()

                                    # gap = np.min(nbr_dist, axis=1).groupby(level=0).min()
                                    # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                                    # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                                    # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                                    # # print(gap.shape)
                                    # # list_gap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[div_ind, all_stm_unique != trial_type, 1]
                                    # list_gap_RRneuron2[slope_ind, div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]
                    print(f'sess_ind {sess_ind}, sampling_ind {sampling_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'overlap_nbr_stimpairs_consis_ABO_sam_cpu_' + str(sess_ind) +  '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_overlap_asis2', 'list_overlap_RRneuron3'],
                     'list_overlap_asis2': list_overlap_asis2, 'list_overlap_RRneuron3': list_overlap_RRneuron3}, f)
                
    print("Ended Process", c_proc.name)

# %%
# randomly sample trials and compute overlap consistency
def compute_overlap_stimpairs_consis_gpu(sess_ind, n_trial_sampling=10):
    print(f"Running on GPU, session {sess_ind}")
    
    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    num_sampling = 10 
    list_num_trials = [10, 25]
    dist_thres = 0.05
    
    np.random.seed(0)

    rate = list_rate_all[sess_ind]
    stm = rate.columns
    num_neurons = rate.shape[0]

    # Multiply by delta t to convert to spike counts
    rate = rate * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
    
    # Convert rate data to a 3D Array
    # Do this indexing on CPU once because it uses complex pandas .loc
    min_num_trials = np.min(all_stm_counts)
    rate_sorted_all_cpu = np.zeros((num_neurons, num_trial_types, min_num_trials))
    rate_copy = rate.copy()
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        rate_sorted_all_cpu[:, trial_type_ind, :] = rate_copy.loc[:, trial_type].iloc[:, :min_num_trials].values

    # Move Main Data to GPU
    rate_sorted_all = cp.asarray(rate_sorted_all_cpu) # deepcopy
    
    # Calculate Mean & Var on GPU
    rate_sorted_mean_coll = cp.mean(rate_sorted_all, axis=2)
    rate_sorted_var_coll = cp.var(rate_sorted_all, axis=2, ddof=1)
    rate_sorted_mean = cp.repeat(rate_sorted_mean_coll[:, :, cp.newaxis], min_num_trials, axis=2)
    rate_sorted_var = cp.repeat(rate_sorted_var_coll[:, :, cp.newaxis], min_num_trials, axis=2)

    list_slopes_dr = cp.asarray(list_slopes_all_an_loglog[sess_ind])

    # Pre-generate Indices (CPU)
    # Generating indices is fast on CPU and easier to manage with RNG
    list_neu_div_inds2 = []
    for sampling_ind in range(num_sampling):
        # Partition neurons
        neu_inds_permuted = np.random.permutation(range(rate_sorted_all.shape[0]))
        neu_div_inds1 = neu_inds_permuted[:int(rate_sorted_all.shape[0]/2)] # 5:5 partitioning
        neu_div_inds2 = neu_inds_permuted[int(rate_sorted_all.shape[0]/2):]
        if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
            neu_div_inds2_copy = neu_div_inds2.copy()
            neu_div_inds2 = neu_div_inds2_copy[:-1]
        list_neu_div_inds2.append((cp.asarray(neu_div_inds1), cp.asarray(neu_div_inds2)))

    for trial_type_ind in range(num_trial_types):
        perm_idx = cp.asarray(np.random.choice(range(min_num_trials), min_num_trials, replace=False))
        rate_sorted_all[:, trial_type_ind, :] = rate_sorted_all[:, trial_type_ind, perm_idx]

    list_rand_trial_inds2 = []
    for num_trials in list_num_trials:
        list_rand_trial_inds = np.zeros((n_trial_sampling, num_trials), dtype=int)
        for t_sam_ind in range(n_trial_sampling):
            list_rand_trial_inds[t_sam_ind] = np.random.choice(range(min_num_trials), num_trials, replace=False)
        list_rand_trial_inds2.append(cp.asarray(list_rand_trial_inds))

    # Result Containers (Initialize on CPU, fill from GPU)
    # Use numpy for storage to save GPU memory, only keeping active calc on GPU
    list_overlap_asis2 = np.full((num_sampling, 2, len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    list_overlap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    tt_offsets = cp.arange(num_trial_types) * min_num_trials
    for sampling_ind in range(num_sampling):
        neu_div_inds1, neu_div_inds2 = list_neu_div_inds2[sampling_ind]
        list_neu_div_inds = [neu_div_inds1, neu_div_inds2]
        for div_ind in range(2):
            neu_idx = list_neu_div_inds[div_ind]
            rate_sorted_2d = rate_sorted_all.reshape(num_neurons, -1)
            sub_data = rate_sorted_2d[neu_idx, :]

            # THE GPU SPEEDUP
            # Calculate the FULL Distance Matrix for all trials vs all trials at once.            
            # full_dist_mat = cp_cdist(sub_data.T, sub_data.T, metric='euclidean')
            full_dist_mat = cdist_euc_gpu(sub_data.T, sub_data.T)
            # cp.fill_diagonal(full_dist_mat, cp.nan)

            # as-is
            for t_ind, num_trials in enumerate(list_num_trials):
                list_rand_trial_inds = list_rand_trial_inds2[t_ind]

                for t_sam_ind in range(n_trial_sampling):
                    local_inds = list_rand_trial_inds[t_sam_ind]
                    global_inds = (tt_offsets[:, cp.newaxis] + local_inds[cp.newaxis, :]).flatten()
                    sampled_mat = full_dist_mat[global_inds][:, global_inds]
                    mat_4d = sampled_mat.reshape(num_trial_types, num_trials, num_trial_types, num_trials)
                    
                    self_blocks = mat_4d.diagonal(axis1=0, axis2=2).transpose(2, 0, 1) # num_trial_types x num_trials x num_trials
                    self_blocks_copy = self_blocks.copy()
                    self_blocks_copy[:, cp.arange(num_trials), cp.arange(num_trials)] = cp.nan
                    self_flat = self_blocks_copy.reshape(num_trial_types, num_trials*num_trials)
                    
                    sorted_self = cp.sort(self_flat, axis=1) # move nan to the last of array
                    n_valid = cp.sum(~cp.isnan(sorted_self), axis=1)
                    # target_idx = (dist_thres * n_valid).astype(int)
                    # thres = sorted_self[cp.arange(num_trial_types), target_idx].reshape(1, 1, num_trial_types)

                    # linear interpolation to mimic np.nanpercentile
                    idx_float = dist_thres * (n_valid - 1)
                    idx_floor = cp.floor(idx_float).astype(int)
                    idx_ceil = cp.ceil(idx_float).astype(int)
                    d = idx_float - idx_floor
                    val_floor = sorted_self[cp.arange(num_trial_types), idx_floor]
                    val_ceil = sorted_self[cp.arange(num_trial_types), idx_ceil]
                    thresh_val = ((1 - d) * val_floor + d * val_ceil)
                    thres = thresh_val.reshape(1, 1, num_trial_types)

                    min_dists = cp.min(mat_4d, axis=3)
                    overlap = cp.mean(min_dists <= (thres + 1e-9), axis=1).T # transpose to num_reference x num_query
                    list_overlap_asis2[sampling_ind, div_ind, t_ind, t_sam_ind] = overlap.get()
                    
                    # for trial_type_ind in range(num_trial_types):
                    #     start_idx = trial_type_ind * num_trials
                    #     end_idx = (trial_type_ind + 1) * num_trials
                    #     self_dist_block = full_dist_mat[start_idx:end_idx, start_idx:end_idx].copy()
                        
                    #     thresh_5 = cp.percentile(self_dist_block[~cp.isnan(self_dist_block)], 5)
                    #     # thresh_mean = cp.nanmean(self_dist_block)
                        
                    #     rest_vs_self_dists = full_dist_mat[:, start_idx:end_idx].copy()
                    #     min_dists = cp.min(rest_vs_self_dists, axis=1)
                    #     is_close = (min_dists <= thresh_5)
                    #     is_close_grouped = is_close.reshape(-1, num_trials)
                    #     overlap = cp.mean(is_close_grouped, axis=1)
                        
                    #     # Use .get() to move the array from GPU to CPU for storage
                    #     list_overlap_asis2[sampling_ind, div_ind, t_ind, t_sam_ind, trial_type_ind] = overlap.get()

        # Handle 0 mean/var (replace with NaN)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = cp.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = cp.nan

        # RRneuron
        # Pre-calculate the common part
        log_mean = cp.log10(rate_sorted_mean_coll)
        mean_log_mean = cp.nanmean(log_mean, axis=0)

        for slope_ind, target_slope in enumerate(list_target_slopes):
            start_time = time()
            
            exponent = (list_slopes_dr[0, :] - target_slope) * mean_log_mean + list_slopes_dr[1, :]
            offset = cp.power(10, exponent)
            
            exponent_noisy = (cp.log10(rate_sorted_var_coll) - list_slopes_dr[1, :]) / list_slopes_dr[0, :] * target_slope + cp.log10(offset)
            var_rs_noisy_coll = cp.power(10, exponent_noisy)
            var_rs_noisy = cp.repeat(var_rs_noisy_coll[:, :, cp.newaxis], min_num_trials, axis=2)
            
            rate_sorted_resid_dr = rate_sorted_all - rate_sorted_mean
            rate_resid_RRneuron_dr = rate_sorted_resid_dr / cp.sqrt(rate_sorted_var) * cp.sqrt(var_rs_noisy)
            rate_RRneuron_dr_all = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr_all = cp.nan_to_num(rate_RRneuron_dr_all, nan=0.0)
            rate_RRneuron_2d = rate_RRneuron_dr_all.reshape(num_neurons, -1)

            for div_ind in range(2):
                neu_idx = list_neu_div_inds[div_ind]
                sub_data = rate_RRneuron_2d[neu_idx, :]
                # full_dist_mat = cp_cdist(sub_data.T, sub_data.T, metric='euclidean')
                full_dist_mat = cdist_euc_gpu(sub_data.T, sub_data.T)
                # cp.fill_diagonal(full_dist_mat, cp.nan)

                for t_ind, num_trials in enumerate(list_num_trials):
                    list_rand_trial_inds = list_rand_trial_inds2[t_ind]
                    
                    for t_sam_ind in range(n_trial_sampling):
                        local_inds = list_rand_trial_inds[t_sam_ind]
                        global_inds = (tt_offsets[:, cp.newaxis] + local_inds[cp.newaxis, :]).flatten()
                        sampled_mat = full_dist_mat[global_inds][:, global_inds]
                        mat_4d = sampled_mat.reshape(num_trial_types, num_trials, num_trial_types, num_trials)
                        
                        self_blocks = mat_4d.diagonal(axis1=0, axis2=2).transpose(2, 0, 1) # num_trial_types x num_trials x num_trials
                        self_blocks_copy = self_blocks.copy()
                        self_blocks_copy[:, cp.arange(num_trials), cp.arange(num_trials)] = cp.nan
                        self_flat = self_blocks_copy.reshape(num_trial_types, num_trials*num_trials)
                        
                        sorted_self = cp.sort(self_flat, axis=1) # move nan to the last of array
                        n_valid = cp.sum(~cp.isnan(sorted_self), axis=1)
                        # target_idx = (dist_thres * n_valid).astype(int)
                        # thres = sorted_self[cp.arange(num_trial_types), target_idx].reshape(1, 1, num_trial_types)
                        
                        # linear interpolation to mimic np.nanpercentile
                        idx_float = dist_thres * (n_valid - 1)
                        idx_floor = cp.floor(idx_float).astype(int)
                        idx_ceil = cp.ceil(idx_float).astype(int)
                        d = idx_float - idx_floor
                        val_floor = sorted_self[cp.arange(num_trial_types), idx_floor]
                        val_ceil = sorted_self[cp.arange(num_trial_types), idx_ceil]
                        thresh_val = ((1 - d) * val_floor + d * val_ceil)
                        thres = thresh_val.reshape(1, 1, num_trial_types)

                        min_dists = cp.min(mat_4d, axis=3)
                        overlap = cp.mean(min_dists <= (thres + 1e-9), axis=1).T # transpose to num_reference x num_query
                        list_overlap_RRneuron3[sampling_ind, slope_ind, div_ind, t_ind, t_sam_ind] = overlap.get()
                        
                        # for trial_type_ind in range(num_trial_types):
                        #     start_idx = trial_type_ind * num_trials
                        #     end_idx = (trial_type_ind + 1) * num_trials
                        #     self_dist_block = full_dist_mat[start_idx:end_idx, start_idx:end_idx].copy()
                        #     cp.fill_diagonal(self_dist_block, cp.nan)
                        #     thresh_5 = cp.percentile(self_dist_block[~cp.isnan(self_dist_block)], 5)
                            
                        #     # rest_mask = cp.ones(full_dist_mat.shape[0], dtype=bool)
                        #     # rest_mask[start_idx:end_idx] = False
                            
                        #     # rest_vs_self_dists = full_dist_mat[rest_mask, :][:, start_idx:end_idx].copy()
                        #     rest_vs_self_dists = full_dist_mat[:, start_idx:end_idx].copy()
                        #     min_dists = cp.min(rest_vs_self_dists, axis=1)
                        #     is_close = (min_dists <= thresh_5)
                        #     is_close_grouped = is_close.reshape(-1, num_trials)
                        #     overlap = cp.mean(is_close_grouped, axis=1)
                            
                        #     list_overlap_RRneuron3[sampling_ind, slope_ind, div_ind, t_ind, t_sam_ind, trial_type_ind] = overlap.get()
            
            print(f"sess_ind {sess_ind}, sampling_ind {sampling_ind}, target_slope {target_slope:.1f}, duration {((time() - start_time)/60):.2f} min")

    # Save
    filename = 'overlap_nbr_stimpairs_consis_ABO_sam_' + str(sess_ind) +  '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_overlap_asis2', 'list_overlap_RRneuron3'],
                     'list_overlap_asis2': list_overlap_asis2, 'list_overlap_RRneuron3': list_overlap_RRneuron3}, f)

    print(f"Ended Process for session {sess_ind}")

# %%
# randomly sample trials and compute overlap
def compute_overlap_stimpairs_trials(sess_ind, n_trial_sampling=10):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    list_num_trials = [10, 25]
    list_num_trials = [30, 35, 40, 45]
    list_num_trials = [10, 25, 40]

    rng = np.random.default_rng(sess_ind)

    print(f'sess_ind: {sess_ind}')
    
    rate = list_rate_all[sess_ind].copy()
    stm = rate.columns.copy()
    num_neurons = rate.shape[0]

    # Multiply by delta t to convert to spike counts
    rate = rate * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # convert to 3D response matrix
    min_num_trials = np.min(all_stm_counts) # session 4 has heterogeneous numbers of trials (minimum 47)

    list_rate_tt = [None] * num_trial_types
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

    rate = np.stack(list_rate_tt, axis=2)
    rate_sorted_all = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x min_num_trials

    # Compute mean & variance for each stimulus
    rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted_all, axis=2), np.var(rate_sorted_all, axis=2, ddof=1)
    rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
        np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)

    list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

    # trial order re-randomization
    for trial_type_ind in range(num_trial_types):
        rate_sorted_all[:, trial_type_ind, :] = rate_sorted_all[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

    # trial sampling
    list_rand_trial_inds2 = np.empty(len(list_num_trials), dtype=object)
    for t_ind, num_trials in enumerate(list_num_trials):
        list_rand_trial_inds = np.zeros((n_trial_sampling, num_trials), dtype=int)
        for t_sam_ind in range(n_trial_sampling):
            list_rand_trial_inds[t_sam_ind] = rng.choice(range(min_num_trials), num_trials, replace=False)
        list_rand_trial_inds2[t_ind] = list_rand_trial_inds.copy()

    list_overlap_asis = np.full((len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    list_gap_asis = np.full((len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    for t_ind, num_trials in enumerate(list_num_trials):
        for t_sam_ind in range(n_trial_sampling):
            print(f'sess_ind {sess_ind}, t_ind {t_ind}, t_sam_ind {t_sam_ind}')
            rate_sorted = rate_sorted_all[:, :, list_rand_trial_inds2[t_ind][t_sam_ind]].copy()
            rate_sorted = pd.DataFrame(rate_sorted.reshape(num_neurons, -1), columns=np.repeat(all_stm_unique, num_trials))

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
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                n_neighbors = 5
                # n_neighbors = all_stm_counts[trial_type_ind]
                nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                
                rate_tt = rate_sorted.loc[:, trial_type].copy()
                rate_rest = rate_sorted.loc[:, all_stm_unique[all_stm_unique != trial_type]].copy()
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
                list_overlap_asis[t_ind, t_sam_ind, trial_type_ind] = overlap.copy()

                # gap = np.min(nbr_dist, axis=1).groupby(level=0).quantile(0.05)
                # # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                # # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().quantile(0.05, axis=1)
                # # print(gap.shape)
                # # list_gap_asis[trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[all_stm_unique != trial_type, 1]
                # list_gap_asis[t_ind, t_sam_ind, trial_type_ind] = gap/list_pwdist[:, 2]

    # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
    rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
    rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
    
    list_overlap_RRneuron2 = np.zeros((len(list_target_slopes), len(list_num_trials), n_trial_sampling, len(list(combinations(range(num_trial_types), 2))), num_neurons, 2))
    list_overlap_RRneuron2 = np.full((len(list_target_slopes), len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    list_gap_RRneuron2 = np.full((len(list_target_slopes), len(list_num_trials), n_trial_sampling, num_trial_types, num_trial_types), np.nan)
    # list_overlap_RRneuron2 = np.full((len(list_target_slopes), len(list(combinations(range(num_trial_types), 2)))), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        # print(f'target slope = {target_slope:.1f}')

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
        rate_sorted_resid_dr = rate_sorted_all - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
            * np.sqrt(var_rs_noisy)
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr_all = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr_all[np.isnan(rate_RRneuron_dr_all)] = 0 # convert NaN to 0!

        start_time = time()
        for t_ind, num_trials in enumerate(list_num_trials):
            for t_sam_ind in range(n_trial_sampling):
                if t_sam_ind % 5 == 0:
                    print(f'sess_ind {sess_ind}, target_slope {target_slope:.1f}, t_ind {t_ind}, t_sam_ind {t_sam_ind}')
                rate_RRneuron_dr = rate_RRneuron_dr_all[:, :, list_rand_trial_inds2[t_ind][t_sam_ind]].copy()
                rate_RRneuron_dr = pd.DataFrame(rate_RRneuron_dr.reshape(num_neurons, -1), columns=np.repeat(all_stm_unique, num_trials))

                # Determine criteria using internal pairwise distance for each stimulus
                list_pwdist = np.zeros((num_trial_types, 3))
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    pwdist = cdist(rate_RRneuron_dr.loc[:, trial_type].T, rate_RRneuron_dr.loc[:, trial_type].T, 'euclidean')
                    pwdist[np.diag_indices(rate_RRneuron_dr.loc[:, trial_type].shape[1])] = np.nan
                    list_pwdist[trial_type_ind, 0] = np.nanpercentile(pwdist, pwdist_thr)
                    list_pwdist[trial_type_ind, 1] = np.nanpercentile(pwdist, 100-pwdist_thr)
                    list_pwdist[trial_type_ind, 2] = np.nanmean(pwdist)

                list_bool_close = np.empty((num_trial_types, num_trial_types), dtype=object)
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    n_neighbors = 5
                    # n_neighbors = all_stm_counts[trial_type_ind]
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                    
                    rate_tt = rate_RRneuron_dr.loc[:, trial_type].copy()
                    rate_rest = rate_RRneuron_dr.loc[:, all_stm_unique[all_stm_unique != trial_type]].copy()
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
                    list_overlap_RRneuron2[slope_ind, t_ind, t_sam_ind, trial_type_ind] = overlap.copy()
                            
                    # gap = np.min(nbr_dist, axis=1).groupby(level=0).quantile(0.05)
                    # # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                    # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                    # # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().quantile(0.05, axis=1)
                    # # print(gap.shape)
                    # # list_gap_RRneuron2[slope_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[all_stm_unique != trial_type, 1]
                    # list_gap_RRneuron2[slope_ind, trial_type_ind] = gap/list_pwdist[:, 2]

        print(f'sess_ind {sess_ind}, target_slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'overlap_nbr_stimpairs_ABO_sam_' + str(sess_ind) +  '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        # pickle.dump({'tree_variables': ['list_overlap_asis', 'list_gap_asis'],
        #              'list_overlap_asis': list_overlap_asis, 'list_gap_asis': list_gap_asis}, f)
        pickle.dump({'tree_variables': ['list_overlap_asis', 'list_overlap_RRneuron2'],
                     'list_overlap_asis': list_overlap_asis, 'list_overlap_RRneuron2': list_overlap_RRneuron2}, f)
                
    print("Ended Process", c_proc.name)

# %%
# overlap of stimulus pairs
def compute_overlap_stimpairs_divneu(sess_ind):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_trial_types = 119

    print(f'sess_ind: {sess_ind}')

    rate = list_rate_all[sess_ind].copy()
    rate_sorted_all = rate.sort_index(axis=1)
    stm = rate_sorted_all.columns.copy()
    num_neurons = rate_sorted_all.shape[0]

    # Multiply by delta t to convert to spike counts
    rate_sorted_all = rate_sorted_all * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted_all)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted_all)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()

    # divide neurons based on spontaneous FF per neuron
    list_neu_div_inds = list_neu_div_inds2[sess_ind]

    list_overlap_asis2 = np.full((len(list_neu_div_inds), num_trial_types, num_trial_types), np.nan)
    list_size_scc_asis = np.full(len(list_neu_div_inds), np.nan)
    for div_ind, neu_div_inds in enumerate(list_neu_div_inds):
        rate_sorted = rate_sorted_all.iloc[neu_div_inds].copy()

        # Determine criteria using internal pairwise distance for each stimulus
        pwdist_thr = 5
        list_pwdist = np.zeros((num_trial_types, 3))
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            pwdist = cdist(rate_sorted.loc[:, trial_type].T, rate_sorted.loc[:, trial_type].T, 'euclidean')
            pwdist[np.diag_indices(rate_sorted.loc[:, trial_type].shape[1])] = np.nan
            list_pwdist[trial_type_ind, 0] = np.nanpercentile(pwdist, pwdist_thr)
            list_pwdist[trial_type_ind, 1] = np.nanpercentile(pwdist, 100-pwdist_thr)
            list_pwdist[trial_type_ind, 2] = np.nanmean(pwdist)

        list_overlap_asis = np.full((num_trial_types, num_trial_types), np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            n_neighbors = 5
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            
            rate_tt = rate_sorted.loc[:, trial_type].copy()
            rate_rest = rate_sorted.loc[:, all_stm_unique[all_stm_unique != trial_type]].copy()
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
        list_overlap_asis2[div_ind] = list_overlap_asis
        
        # strongly connected component (SCC)

        # convert overlap matrix digonal to 0
        list_overlap_asis[np.eye(num_trial_types, dtype=bool)] = 0

        G_dir = nx.from_numpy_array(list_overlap_asis, create_using=nx.DiGraph)
        sccs = nx.strongly_connected_components(G_dir)
        largest_scc = max(sccs, key=len)
        G_scc = G_dir.subgraph(largest_scc).copy()
        size_scc_asis = len(list(G_scc.nodes))
        list_size_scc_asis[div_ind] = size_scc_asis

    # Save into a file
    filename = 'overlap_nbr_stimpairs_ABO_divneu_' + str(sess_ind) +  '.pickle.gz'
    with gzip.open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_overlap_asis2', 'list_size_scc_asis'],
                     'list_overlap_asis2': list_overlap_asis2, 'list_size_scc_asis': list_size_scc_asis}, f)

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
    list_slopes_all_att_loglog = resp_matrix_ep_RS_all['list_slopes_all_att_loglog'].copy()
    list_slopes_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_all_an_loglog'].copy()

# ABO higher visual areas
with gzip.open('resp_matrix_ep_HVA_allensdk.pickle.gz', 'rb') as f:
    resp_matrix_ep_HVA_allensdk = pickle.load(f)

    list_rate_all_HVA = dc(resp_matrix_ep_HVA_allensdk['list_rate_all_HVA'])
    list_slopes_all_an_loglog_HVA = dc(resp_matrix_ep_HVA_allensdk['list_slopes_all_an_loglog_HVA'])
    list_empty_sess2 = dc(resp_matrix_ep_HVA_allensdk['list_empty_sess2'])
    list_num_neurons_HVA = dc(resp_matrix_ep_HVA_allensdk['list_num_neurons_HVA'])

with gzip.open('resp_matrix_ep_naturalmovie_FC_allensdk.pickle.gz', 'rb') as f:
    resp_matrix_ep_naturalmovie = pickle.load(f)

    brain_observatory_sessid = resp_matrix_ep_naturalmovie['brain_observatory_sessid'].copy()
    list_sess_ids = resp_matrix_ep_naturalmovie['list_sess_ids'].copy()

with gzip.open('resp_matrix_ep_spt_BO_all_32sess_gpu.pickle.gz', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_spt_BO_all = resp_matrix_ep_RS_all['list_rate_spt_BO_all'].copy()

with gzip.open('unit_ids.pickle.gz', 'rb') as f:
    unit_ids_load = pickle.load(f)

    list_unit_ids_visp = unit_ids_load['list_unit_ids_visp'].copy()
    list_sess_data = unit_ids_load['list_sess_data'].copy()

# receptive field metrics for each V1 unit
save_file_name = 'unit_rf_metrics_all.pickle.gz'
with gzip.open(save_file_name, 'rb') as f:
    unit_rf_metrics_all = pickle.load(f)
    list_rfmet2 = unit_rf_metrics_all['list_rfmet2'].copy()
    list_num_neurons_lr = unit_rf_metrics_all['list_num_neurons_lr'].copy()
    list_num_neurons_ud = unit_rf_metrics_all['list_num_neurons_ud'].copy()
    list_slopes_all_an_loglog_onscreen = unit_rf_metrics_all['list_slopes_all_an_loglog_onscreen'].copy()
    list_slopes_all_an_loglog_onscreen_real = unit_rf_metrics_all['list_slopes_all_an_loglog_onscreen_real'].copy()
list_rfmet2 = list_rfmet2[np.isin(list_sess_ids, brain_observatory_sessid)]
list_num_neurons_lr = list_num_neurons_lr[np.isin(list_sess_ids, brain_observatory_sessid)]
list_num_neurons_ud = list_num_neurons_ud[np.isin(list_sess_ids, brain_observatory_sessid)]

with gzip.open('invalid_unit_sess.pickle.gz', 'rb') as f:
    invalid_unit_sess = pickle.load(f)

    list_invalid_units_visp = invalid_unit_sess['list_invalid_units_visp'].copy()
    list_invalid_units_HVA = invalid_unit_sess['list_invalid_units_HVA'].copy()
    list_invalid_sess_visp = invalid_unit_sess['list_invalid_sess_visp'].copy()
    list_invalid_sess_HVA = invalid_unit_sess['list_invalid_sess_HVA'].copy()
    sess_inds_mapBO = invalid_unit_sess['sess_inds_mapBO'].copy()

save_file_name = 'slopes_divneu_all.pickle.gz' # intercept per neuron
save_file_name = 'slopes_divneu_all2.pickle.gz' # spontaneous FF
with gzip.open(save_file_name, 'rb') as f:
    slopes_divneu_all = pickle.load(f)
    list_slopes_all_an_loglog_divneu = slopes_divneu_all['list_slopes_all_an_loglog_divneu'].copy()
    list_neu_div_inds2 = slopes_divneu_all['list_neu_div_inds2'].copy()

# number of neurons for each session
num_sess = 32
list_num_neurons_visp = np.zeros(num_sess, dtype=int)
for sess_ind in range(num_sess):
    list_num_neurons_visp[sess_ind] = list_rate_all[sess_ind].shape[0]

# %%
# multiprocessing
list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
# num_sess = len(list_sess_ids)
num_sess = 32

# unit receptive field metrics
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool: # memory-heavy
        list_inputs = [[sess_ind] for sess_ind, sess_id in enumerate(list_sess_ids) if np.isin(sess_id, brain_observatory_sessid)]
        
        pool.starmap(unit_rfmet, list_inputs)

# decoding
decoder_type = 'SVM'
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind, decoder_type] for sess_ind in range(num_sess)]
        
        pool.starmap(decode_divneu, list_inputs)

# RSA
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(RSA_across_sesspairs_ABO_rf, list_inputs)

# RSA
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind, 'cos_sim'] for sess_ind in range(num_sess)]
        
        pool.starmap(RSA_across_sesspairs_ABO_divneu, list_inputs)

# RSA
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(RSA_withinsess_ABO_rf, list_inputs)

# RSA
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind, 'cos_sim'] for sess_ind in range(num_sess)]
        
        pool.starmap(RSA_withinsess_ABO_divneu, list_inputs)

# RSA
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(RSA_withinsess_ABO_vis, list_inputs)

# effective dimensionality
n_trial_sampling = 100
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, n_trial_sampling] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_eff_dim, list_inputs)

# slope for different neuronal groups
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind] for sess_ind in range(num_sess)]
        
        pool.starmap(fit_slopes_divneu, list_inputs)

# overlap (trial sampling)
n_trial_sampling = 10
if __name__ == '__main__':
    
    # with mp.Pool() as pool:
    #     list_inputs = [[sess_ind, n_trial_sampling] for sess_ind in range(num_sess)]
        
    #     pool.starmap(compute_overlap_stimpairs_consis, list_inputs)

    for sess_ind in range(num_sess):
        compute_overlap_stimpairs_consis_gpu(sess_ind, n_trial_sampling)

# overlap (trial sampling)
n_trial_sampling = 10
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, n_trial_sampling] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_overlap_stimpairs_trials, list_inputs)

# overlap (divide neurons)
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_overlap_stimpairs_divneu, list_inputs)
        