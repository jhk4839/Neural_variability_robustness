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

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, norm, kruskal, tukey_hsd, mode, spearmanr, rankdata, nbinom, poisson, binom
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

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as logit
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, AffinityPropagation, KMeans
from sklearn.decomposition import PCA

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
# decoding (Ecker)
def decode_Ecker(sess_ind, decoder_type):

    ''' decoder_type is SVM, logit, RF, kNN '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        n_splits = 10
        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
        
        # print(f'sess_ind: {sess_ind}')

        rate = list_rate_pooled_neutrialresamp[sess_ind].copy()
        rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()
        
        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        all_stimuli = all_stm_unique.copy()
        train_stimuli = all_stimuli.copy()
        # n_samples = np.round(np.mean(all_stm_counts)).astype(int)
        
        # # convert to 3D response matrix
        # min_num_trials = np.min(all_stm_counts)

        # list_rate_tt = [None] * num_trial_types
        # for trial_type_ind, trial_type in enumerate(all_stimuli):
        #     list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        # rate = np.stack(list_rate_tt, axis=2)
        # rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x min_num_trials

        # Compute mean & variance for each stimulus
        rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
        rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

        list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog_pooled_resampraw[sess_ind], columns=rate_sorted_mean_coll.columns).copy()
        
        # decoding cross-validation (as-is)
        stkfold = StratifiedKFold(n_splits=n_splits)

        # Re-convert to 2D response matrix
        label_train = rate_sorted.columns.copy()
        rate_train = rate_sorted.copy()

        list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
        list_accuracy = np.full(n_splits, np.nan)

        for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
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
            list_confusion_test[split_ind] = test_confusion_matrix.copy()

            accuracy = accuracy_score(y_test, y_test_pred)
            list_accuracy[split_ind] = accuracy
            # print(accuracy)

        # calculate cross-validation average test confusion matrix/test accuracy
        mean_confusion_test_asis = sum(list_confusion_test) / n_splits
        mean_confusion_test_asis = pd.DataFrame(mean_confusion_test_asis, columns=train_stimuli, index=train_stimuli).fillna(0)
        # print(mean_confusion_test.round(3))
        
        mean_accuracy_asis = np.mean(list_accuracy)
        # print(round(mean_accuracy, ndigits=3))

        # RRneuron
        list_mean_confusion_test_RRneuron = np.full((len(list_target_slopes), len(train_stimuli), len(train_stimuli)), np.nan)
        list_mean_accuracy_RRneuron = np.full(len(list_target_slopes), np.nan)
        for slope_ind, target_slope in enumerate(list_target_slopes):
            start_time = time()

            # print(f'sess_ind: {sess_ind}, target slope {target_slope:.1f}')
                                
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
                        
            # decoding cross-validation (RRneuron)  

            # Re-convert to 2D response matrix
            label_train_RRneuron = rate_RRneuron_dr.columns.copy()
            rate_train_RRneuron = rate_RRneuron_dr.copy()
            
            # decoding cross-validation (as-is)
            stkfold = StratifiedKFold(n_splits=n_splits)

            list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
            list_accuracy = np.full(n_splits, np.nan)

            for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
                X_train, X_test = rate_train_RRneuron.T.iloc[train_index].copy(), rate_train_RRneuron.T.iloc[test_index].copy() # train, test data/label
                y_train, y_test = label_train_RRneuron[train_index].copy(), label_train_RRneuron[test_index].copy()

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
                list_confusion_test[split_ind] = test_confusion_matrix.copy()

                accuracy = accuracy_score(y_test, y_test_pred)
                list_accuracy[split_ind] = accuracy
                # print(accuracy)

            print(f'session {sess_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

            # calculate cross-validation average test confusion matrix/test accuracy
            mean_confusion_test = sum(list_confusion_test) / n_splits
            mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
            # print(mean_confusion_test_Bayes.round(3))
            list_mean_confusion_test_RRneuron[slope_ind] = mean_confusion_test.copy()
            
            mean_accuracy = np.mean(list_accuracy)
            # print(round(mean_accuracy, ndigits=3))
            list_mean_accuracy_RRneuron[slope_ind] = mean_accuracy

            # print(f'sess_ind: {sess_ind}, rescale r {rf}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = decoder_type + '_decoding_Ecker_allstim_pooled_neutrialresamp_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['mean_confusion_test_asis', 'mean_accuracy_asis', 'list_mean_confusion_test_RRneuron', 'list_mean_accuracy_RRneuron'],
                     'mean_confusion_test_asis': mean_confusion_test_asis, 'mean_accuracy_asis': mean_accuracy_asis,
                     'list_mean_confusion_test_RRneuron': list_mean_confusion_test_RRneuron, 'list_mean_accuracy_RRneuron': list_mean_accuracy_RRneuron}, f)
                        
    print("Ended Process", c_proc.name)

# %%
# RSA across session pairs (Ecker Neuropixels, RRneuron)

def RSA_across_sesspairs_Ecker(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
        num_trial_types = 75

        print(f'sess_ind {sess_ind}')

        rng = np.random.default_rng(sess_ind) # match trial order.

        rate = list_rate_pooled_neutrialresamp[sess_ind].copy()
        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # convert to 3D response matrix
        min_num_trials = np.min(all_stm_counts)
        max_num_trials = np.max(all_stm_counts)

        # trial order re-randomization
        list_rand_trial_inds = np.empty(num_trial_types, dtype=object)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            rate_tt = rate.loc[:, trial_type]
            list_rand_trial_inds[trial_type_ind] = rng.permutation(np.arange(rate_tt.shape[1]))
            rate.loc[:, trial_type] = rate_tt.iloc[:, list_rand_trial_inds[trial_type_ind]]

        rate_sorted = np.full((rate.shape[0], num_trial_types, max_num_trials), np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            rate_sorted[:, trial_type_ind, :all_stm_counts[trial_type_ind]] = rate.loc[:, trial_type] # NaNs are at the end

        rate_sorted_mean_coll, rate_sorted_var_coll = np.nanmean(rate_sorted, axis=2), np.nanvar(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.full_like(rate_sorted, np.nan), np.full_like(rate_sorted, np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            rate_mean_tt = rate_sorted_mean_coll[:, trial_type_ind]
            rate_sorted_mean[:, trial_type_ind, :all_stm_counts[trial_type_ind]] = np.repeat(rate_mean_tt[:, np.newaxis], all_stm_counts[trial_type_ind], axis=1) # NaNs are at the end
            rate_var_tt = rate_sorted_var_coll[:, trial_type_ind]
            rate_sorted_var[:, trial_type_ind, :all_stm_counts[trial_type_ind]] = np.repeat(rate_var_tt[:, np.newaxis], all_stm_counts[trial_type_ind], axis=1) # NaNs are at the end

        list_slopes_dr = list_slopes_all_an_loglog_pooled_resampraw[sess_ind].copy()

        # repeat calculating similarity matrices

        # n_neurons x n_stimuli 2D matrix sampling
        tt_pairs = list(combinations(range(max_num_trials), 2))
        n_sampling = np.min([len(tt_pairs), 10000])
        # n_sampling = len(tt_pairs)

        list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
        
        count = 0
        for sampling_ind in range(n_sampling):
            rate_sampled_trials1 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][0]]).copy()
            rate_sampled_trials2 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][1]]).copy()

            RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2))

            # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos))
            list_RSM[sampling_ind] = RSM.copy()
            
            count += 1
            if count % (n_sampling//2) == 0:
                print(f'count: {count}')

        RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
        list_RSM_mean_asis = RSM_mean.copy()

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
        
        list_RSM_mean_RRneuron = np.zeros((len(list_target_slopes), num_trial_types, num_trial_types))
        list_rate_RRneuron_dr = np.empty(len(list_target_slopes), dtype=object)
        for slope_ind, target_slope in enumerate(list_target_slopes):
            start_time = time()

            # calculate target variance
            var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

            # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed
            offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :])

            var_rs_noisy = \
                pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                    / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
            var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], max_num_trials, axis=2)
            var_rs_noisy[np.isnan(rate_sorted)] = np.nan # convert the beyond-num_trials end to NaN for each stimulus

            # Compute changed residual and add back to the mean
            rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                * np.sqrt(var_rs_noisy)
            # print(rate_resid_RRneuron_dr)
            rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!
            rate_RRneuron_dr[np.isnan(rate_sorted)] = np.nan # convert the beyond-num_trials end to NaN for each stimulus

            # # trial order re-randomization
            # for trial_type_ind in range(num_trial_types):
            #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

            list_rate_RRneuron_dr[slope_ind] = rate_RRneuron_dr.copy()

            # repeat calculating similarity matrices

            # n_neurons x n_stimuli 2D matrix sampling

            # if ind == 3:
            #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
            tt_pairs = list(combinations(range(max_num_trials), 2))
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
                    print(f'count: {count}')

            RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
            list_RSM_mean_RRneuron[slope_ind] = RSM_mean.copy()
            print(f'session {sess_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'RSM_Ecker_allneu_pooled_neutrialresamp_' + similarity_type + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis', 'list_rate_RRneuron_dr', 'list_RSM_mean_RRneuron'], \
                     'list_RSM_mean_asis': list_RSM_mean_asis, 'list_rate_RRneuron_dr': list_rate_RRneuron_dr, 'list_RSM_mean_RRneuron': list_RSM_mean_RRneuron}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA within sessions (Ecker, RRneuron)

def RSA_withinsess_Ecker(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
        num_trial_types = 75
        n_neu_sampling = 10

        print(f'sess_ind {sess_ind}')

        # Iterate over all sessions
        rng = np.random.default_rng(sess_ind) # match neuron partitioning
        # random.seed(0)

        list_corr_withinsess_asis = np.full((n_neu_sampling, 3), np.nan)
        list_corr_withinsess2 = np.full((len(list_target_slopes), n_neu_sampling, 3), np.nan)

        rate = list_rate_pooled_neutrialresamp[sess_ind].copy()
        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # convert to 3D response matrix
        min_num_trials = np.min(all_stm_counts)
        max_num_trials = np.max(all_stm_counts)

        # trial order re-randomization
        list_rand_trial_inds = np.empty(num_trial_types, dtype=object)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            rate_tt = rate.loc[:, trial_type]
            list_rand_trial_inds[trial_type_ind] = rng.permutation(np.arange(rate_tt.shape[1]))
            rate.loc[:, trial_type] = rate_tt.iloc[:, list_rand_trial_inds[trial_type_ind]]

        rate_sorted = np.full((rate.shape[0], num_trial_types, max_num_trials), np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            rate_sorted[:, trial_type_ind, :all_stm_counts[trial_type_ind]] = rate.loc[:, trial_type] # NaNs are at the end

        rate_sorted_mean_coll, rate_sorted_var_coll = np.nanmean(rate_sorted, axis=2), np.nanvar(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.full_like(rate_sorted, np.nan), np.full_like(rate_sorted, np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            rate_mean_tt = rate_sorted_mean_coll[:, trial_type_ind]
            rate_sorted_mean[:, trial_type_ind, :all_stm_counts[trial_type_ind]] = np.repeat(rate_mean_tt[:, np.newaxis], all_stm_counts[trial_type_ind], axis=1) # NaNs are at the end
            rate_var_tt = rate_sorted_var_coll[:, trial_type_ind]
            rate_sorted_var[:, trial_type_ind, :all_stm_counts[trial_type_ind]] = np.repeat(rate_var_tt[:, np.newaxis], all_stm_counts[trial_type_ind], axis=1) # NaNs are at the end
        
        list_slopes_dr = list_slopes_all_an_loglog_pooled_resampraw[sess_ind].copy()

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        list_rate_RRneuron_dr = np.full((len(list_target_slopes), *rate_sorted.shape), np.nan)
        for slope_ind, target_slope in enumerate(list_target_slopes):
            # calculate target variance
            var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

            # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed
            offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :])

            var_rs_noisy = \
                pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                    / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # collapsed
            var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], max_num_trials, axis=2)
            var_rs_noisy[np.isnan(rate_sorted)] = np.nan # convert the beyond-num_trials end to NaN for each stimulus

            # Compute changed residual and add back to the mean
            rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                * np.sqrt(var_rs_noisy)
            # print(rate_resid_RRneuron_dr)
            rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # convert NaN to 0!
            rate_RRneuron_dr[np.isnan(rate_sorted)] = np.nan # convert the beyond-num_trials end to NaN for each stimulus

            # # trial order re-randomization
            # for trial_type_ind in range(num_trial_types):
            #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

            list_rate_RRneuron_dr[slope_ind] = rate_RRneuron_dr.copy()

        # Iterate over neuron partitionings
        tt_pairs = list(combinations(range(max_num_trials), 2))
        # random.shuffle(tt_pairs)
        n_sampling = np.min([len(tt_pairs), 10000])
        # n_sampling = len(tt_pairs)

        for neu_sample_ind in range(n_neu_sampling):
            # print(f'neu_sample_ind = {neu_sample_ind}')
            
            # Partition neurons
            neu_inds_permuted = rng.permutation(range(rate_sorted.shape[0]))
            neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 partitioning
            neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
            if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
                neu_div_inds2 = neu_div_inds2[:-1].copy()

            # as-is

            # repeat calculating similarity matrices
            
            # n_neurons x n_stimuli 2D matrix sampling
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
            list_corr_withinsess_asis[neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_withinsess_asis[neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_withinsess_asis[neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
        
            for slope_ind, target_slope in enumerate(list_target_slopes):
                start_time = time()

                # RRneuron
                rate_RRneuron_dr = list_rate_RRneuron_dr[slope_ind].copy()

                # repeat calculating similarity matrices
                
                # n_neurons x n_stimuli 2D matrix sampling
                
                tt_pairs = list(combinations(range(max_num_trials), 2))
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
                    if count % (n_sampling) == 0:
                        print(f'count: {count}')

                RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
                RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                list_corr_withinsess2[slope_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_withinsess2[slope_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_withinsess2[slope_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
                
                print(f'session {sess_ind}, neu_sample_ind {neu_sample_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'RSM_corr_withinsess_Ecker_pooled_neutrialresamp_' + similarity_type + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis', 'list_corr_withinsess2'], \
                    'list_corr_withinsess_asis': list_corr_withinsess_asis, 'list_corr_withinsess2': list_corr_withinsess2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# Effective dimensionality
def compute_eff_dim(sess_ind, n_trial_sampling=100):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

    num_trial_types = 75
    rng = np.random.default_rng(sess_ind)

    list_dim_asis = np.zeros(num_trial_types)
    list_dim_RRneuron = np.zeros((len(list_target_slopes), num_trial_types))
    list_dim_global_asis = np.zeros(2)
    list_dim_global_RRneuron = np.zeros((len(list_target_slopes), 2))
    list_dim_sam_asis = np.zeros(n_trial_sampling)
    list_dim_sam_RRneuron = np.zeros((len(list_target_slopes), n_trial_sampling))

    print(f'sess_ind: {sess_ind}')

    rate = list_rate_pooled_neutrialresamp[sess_ind].copy()
    rate_sorted = rate.sort_index(axis=1)
    stm = rate.columns.copy()

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog_pooled_resampraw[sess_ind], \
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
    filename = 'eff_dim_DC_Ecker_pooled_neutrialresamp_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_dim_asis', 'list_dim_RRneuron', 'list_dim_global_asis', 'list_dim_global_RRneuron', 'list_dim_sam_asis', 'list_dim_sam_RRneuron'],
                    'list_dim_asis': list_dim_asis, 'list_dim_RRneuron': list_dim_RRneuron, 'list_dim_global_asis': list_dim_global_asis, 'list_dim_global_RRneuron': list_dim_global_RRneuron,
                    'list_dim_sam_asis': list_dim_sam_asis, 'list_dim_sam_RRneuron': list_dim_sam_RRneuron}, f)

    print("Ended Process", c_proc.name)

# %%
# loading variables
with open('resp_matrix_ep_all_Ecker.pickle', 'rb') as f:
    resp_matrix_ep_all = pickle.load(f)

    list_rate_all = resp_matrix_ep_all['list_rate_all'].copy()
    list_slopes_all_an_loglog = resp_matrix_ep_all['list_slopes_all_an_loglog'].copy()
    sess_inds_qual_all = resp_matrix_ep_all['sess_inds_qual_all'].copy()
    subject_ids = resp_matrix_ep_all['subject_ids'].copy()
    
    list_rate_shuf_pooled = resp_matrix_ep_all['list_rate_shuf_pooled'].copy()
    list_slopes_all_an_loglog_pooled = resp_matrix_ep_all['list_slopes_all_an_loglog_pooled'].copy()
    
    list_rate_pooled_resamp = resp_matrix_ep_all['list_rate_pooled_resamp'].copy()
    list_resamp_inds = resp_matrix_ep_all['list_resamp_inds'].copy()
    list_slopes_all_an_loglog_pooled_resamp = resp_matrix_ep_all['list_slopes_all_an_loglog_pooled_resamp'].copy()
    
    list_rate_pooled_neutrialresamp = resp_matrix_ep_all['list_rate_pooled_neutrialresamp'].copy()
    list_slopes_all_an_loglog_pooled_resampraw = resp_matrix_ep_all['list_slopes_all_an_loglog_pooled_resampraw'].copy()

unique_ids = np.unique(subject_ids)
r_temp = [[] for monkey_ind in range(len(unique_ids))]
p_temp = [[] for monkey_ind in range(len(unique_ids))]
list_r_estim_neu2_pooled = np.empty(len(unique_ids), dtype=object)
list_p_estim_neu2_pooled = np.empty(len(unique_ids), dtype=object)
for monkey_ind, monkey_id in enumerate(unique_ids):
    sess_inds = np.where(subject_ids == monkey_id)[0]
    for ind, sess_ind in enumerate(sess_inds):
        r_temp[monkey_ind].append(list_r_estim_neu2[sess_ind])
        p_temp[monkey_ind].append(list_p_estim_neu2[sess_ind])
    list_r_estim_neu2_pooled[monkey_ind] = np.concatenate(r_temp[monkey_ind])
    list_p_estim_neu2_pooled[monkey_ind] = np.concatenate(p_temp[monkey_ind])

# %%
# multiprocessing
num_sess = len(list_rate_pooled_neutrialresamp)

# decoding
decoder_type = 'SVM'
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, decoder_type] for sess_ind in range(num_sess)]
        
        pool.starmap(decode_Ecker, list_inputs)

# RSA
similarity_type = 'cos_sim'
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, similarity_type] for sess_ind in range(num_sess)]
        
        pool.starmap(RSA_across_sesspairs_Ecker, list_inputs)

# RSA
similarity_type = 'cos_sim'
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, similarity_type] for sess_ind in range(num_sess)]
        
        pool.starmap(RSA_withinsess_Ecker, list_inputs)

# effective dimensionality
n_trial_sampling = 100
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, n_trial_sampling] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_eff_dim, list_inputs)
