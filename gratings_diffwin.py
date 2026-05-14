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

# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

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
        trial_mean = np.mean(trial_rate, axis=1)
        trial_var = np.var(trial_rate, axis=1, ddof=1)

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
# decoding (gratings)
def decode_gratings(sess_ind, decoder_type):

    ''' decoder_type is SVM, logit, RF, kNN '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        n_splits = 10
        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
        
        # print(f'sess_ind: {sess_ind}')

        # rate = list_rate_sg_all[sess_ind].copy()
        rate = list_rate_dg75_250_all[sess_ind].copy()
        rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Multiply by delta t to convert to spike counts
        rate_sorted = rate_sorted * 0.25

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        all_stimuli = all_stm_unique.copy()
        train_stimuli = all_stimuli.copy()
        
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

        # list_slopes_dr = pd.DataFrame(list_slopes_sg_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()
        list_slopes_dr = pd.DataFrame(list_slopes_dg75_250_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()
        
        # decoding cross-validation (as-is)
        stkfold = StratifiedKFold(n_splits=n_splits)

        # Re-convert to 2D response matrix
        label_train = rate_sorted.columns.copy()
        rate_train = rate_sorted.copy()

        list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
        list_accuracy = np.full(n_splits, np.nan)

        for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
        # for probe_ind, probe_stim in enumerate(all_stimuli):
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

            # Compute mean and variance of slope-changed data
            rate_mean_RRneuron_coll, rate_var_RRneuron_coll = \
                compute_mean_var_trial_collapse(stm_cnt_dict, rate_RRneuron_dr)
                        
            # decoding cross-validation (RRneuron)  

            # Re-convert to 2D response matrix
            label_train_RRneuron = rate_RRneuron_dr.columns.copy()
            rate_train_RRneuron = rate_RRneuron_dr.copy()

            # decoding cross-validation (as-is)
            stkfold = StratifiedKFold(n_splits=n_splits)

            list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
            list_accuracy = np.full(n_splits, np.nan)

            for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
            # for probe_ind, probe_stim in enumerate(all_stimuli):
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
    filename = decoder_type + '_decoding_sg_allstim_' + str(sess_ind) + '.pickle'
    filename = decoder_type + '_decoding_dg75_250_allstim_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['mean_confusion_test_asis', 'mean_accuracy_asis', 'list_mean_confusion_test_RRneuron', 'list_mean_accuracy_RRneuron'],
                     'mean_confusion_test_asis': mean_confusion_test_asis, 'mean_accuracy_asis': mean_accuracy_asis,
                     'list_mean_confusion_test_RRneuron': list_mean_confusion_test_RRneuron, 'list_mean_accuracy_RRneuron': list_mean_accuracy_RRneuron}, f)
                
    print("Ended Process", c_proc.name)

# %%
# decoding (different time windows)
def decode_diffwin(sess_ind, decoder_type):

    ''' decoder_type is SVM, logit, RF, kNN '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        n_splits = 10
        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
        list_wins = np.arange(50, 250, 50) # time window size
        num_trial_types = 119

        list_mean_confusion_test_asis = np.empty((len(list_wins), num_trial_types, num_trial_types), dtype=object)
        list_mean_accuracy_asis = np.full(len(list_wins), np.nan)
        list_mean_confusion_test_RRneuron2 = np.empty((len(list_wins), len(list_target_slopes), num_trial_types, num_trial_types), dtype=object)
        list_mean_accuracy_RRneuron2 = np.full((len(list_wins), len(list_target_slopes)), np.nan)
        for win_ind, win in enumerate(list_wins):
            if win_ind in [0, 1]:
                print(f'sess_ind: {sess_ind}, time window {win}ms')

                rate = list_rate_diffwin_trunc_all[sess_ind, win_ind].copy()
                # rate_sorted = rate.sort_index(axis=1)
                stm = rate.columns.copy()

                # Create a counting dictionary for each stimulus
                all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
                stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
                all_stimuli = all_stm_unique.copy()
                train_stimuli = all_stimuli.copy()
                
                # convert to 3D response matrix
                min_num_trials = np.min(all_stm_counts)

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

                list_slopes_dr = list_slopes_diffwin_trunc_all_an_loglog[sess_ind, win_ind].copy()

                # trial order re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]
                
                # decoding cross-validation (as-is)
                stkfold = StratifiedKFold(n_splits=n_splits)

                # Re-convert to 2D response matrix
                label_train = np.repeat(all_stimuli, min_num_trials)
                rate_train = pd.DataFrame(rate_sorted.reshape(rate_sorted.shape[0], -1), columns=label_train)

                list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
                list_accuracy = np.full(n_splits, np.nan)

                for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
                # for probe_ind, probe_stim in enumerate(all_stimuli):
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
                list_mean_confusion_test_asis[win_ind] = mean_confusion_test_asis.copy()
                
                mean_accuracy_asis = np.mean(list_accuracy)
                # print(round(mean_accuracy, ndigits=3))
                list_mean_accuracy_asis[win_ind] = mean_accuracy_asis

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
                                
                    # decoding cross-validation (RRneuron)  

                    # Re-convert to 2D response matrix
                    label_train_RRneuron = np.repeat(all_stimuli, min_num_trials)
                    rate_train_RRneuron = pd.DataFrame(rate_RRneuron_dr.reshape(rate_RRneuron_dr.shape[0], -1), columns=label_train_RRneuron)

                    # decoding cross-validation (as-is)
                    stkfold = StratifiedKFold(n_splits=n_splits)

                    list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
                    list_accuracy = np.full(n_splits, np.nan)

                    for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
                    # for probe_ind, probe_stim in enumerate(all_stimuli):
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

                    print(f'session {sess_ind}, time window {win}ms, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

                    # calculate cross-validation average test confusion matrix/test accuracy
                    mean_confusion_test = sum(list_confusion_test) / n_splits
                    mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
                    # print(mean_confusion_test_Bayes.round(3))
                    list_mean_confusion_test_RRneuron[slope_ind] = mean_confusion_test.copy()
                    
                    mean_accuracy = np.mean(list_accuracy)
                    # print(round(mean_accuracy, ndigits=3))
                    list_mean_accuracy_RRneuron[slope_ind] = mean_accuracy

                    # print(f'sess_ind: {sess_ind}, rescale r {rf}, duration {(time()-start_time)/60:.2f} min')
                
                list_mean_confusion_test_RRneuron2[win_ind] = list_mean_confusion_test_RRneuron.copy()
                list_mean_accuracy_RRneuron2[win_ind] = list_mean_accuracy_RRneuron.copy()
            
    # Save into a file
    filename = decoder_type + '_decoding_diffwin_trunc_allstim_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_mean_confusion_test_asis', 'list_mean_accuracy_asis', 'list_mean_confusion_test_RRneuron2', 'list_mean_accuracy_RRneuron2'],
                     'list_mean_confusion_test_asis': list_mean_confusion_test_asis, 'list_mean_accuracy_asis': list_mean_accuracy_asis,
                     'list_mean_confusion_test_RRneuron2': list_mean_confusion_test_RRneuron2, 'list_mean_accuracy_RRneuron2': list_mean_accuracy_RRneuron2}, f)
                
    print("Ended Process", c_proc.name)

# %%
# RSA across session pairs
def RSA_across_sesspairs_gratings(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 8

    rng = np.random.default_rng(sess_ind) # match trial order.

    print(f'sess_ind {sess_ind}')

    # rate = list_rate_sg_all[sess_ind].copy()
    rate = list_rate_dg75_250_all[sess_ind].copy()
    # rate_sorted = rate.sort_index(axis=1)
    stm = rate.columns.copy()

    # Multiply by delta t to convert to spike counts
    rate = rate * 0.25
    
    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
    
    # convert to 3D response matrix
    min_num_trials = np.min(all_stm_counts)

    list_rate_tt = [None] * num_trial_types
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

    rate = np.stack(list_rate_tt, axis=2)
    rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials

    rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
    rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
        np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
    
    # list_slopes_dr = list_slopes_sg_all_an_loglog[sess_ind].copy()
    list_slopes_dr = list_slopes_dg75_250_all_an_loglog[sess_ind].copy()

    # trial order re-randomization
    for trial_type_ind in range(num_trial_types):
        rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

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
    
    list_rate_RRneuron_dr = np.empty(len(list_target_slopes), dtype=object)
    list_RSM_mean_RRneuron = np.zeros((len(list_target_slopes), num_trial_types, num_trial_types))
    for slope_ind, target_slope in enumerate(list_target_slopes):
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
        
        # # trial order re-randomization
        # for trial_type_ind in range(num_trial_types):
        #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

        list_rate_RRneuron_dr[slope_ind] = rate_RRneuron_dr.copy()

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
                print(f'count: {count}')

        RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
        list_RSM_mean_RRneuron[slope_ind] = RSM_mean.copy()

    # Save into a file
    filename = 'RSM_sg_allneu_' + similarity_type + str(sess_ind) + '.pickle'
    filename = 'RSM_dg75_250_allneu_' + similarity_type + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis', 'list_rate_RRneuron_dr', 'list_RSM_mean_RRneuron'], \
                     'list_RSM_mean_asis': list_RSM_mean_asis, 'list_rate_RRneuron_dr': list_rate_RRneuron_dr, 'list_RSM_mean_RRneuron': list_RSM_mean_RRneuron}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA across session pairs (different time windows)
def RSA_across_sesspairs_diffwin(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    list_wins = np.arange(50, 250, 50) # time window size

    rng = np.random.default_rng(sess_ind) # match trial order.
    
    # trial order re-randomization (to match trial order across time windows)
    rate = list_rate_diffwin_trunc_all[sess_ind, 0].copy()
    stm = rate.columns.copy()
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
    min_num_trials = np.min(all_stm_counts)   

    list_rand_trial_inds = np.zeros((num_trial_types, min_num_trials), dtype=int)
    for trial_type_ind in range(num_trial_types):
        list_rand_trial_inds[trial_type_ind] = rng.choice(range(min_num_trials), min_num_trials, replace=False)

    list_RSM_mean_asis2 = np.full((len(list_wins), num_trial_types, num_trial_types), np.nan)
    list_rate_RRneuron_dr2 = np.empty((len(list_wins), len(list_target_slopes)), dtype=object)
    list_RSM_mean_RRneuron2 = np.full((len(list_wins), len(list_target_slopes), num_trial_types, num_trial_types), np.nan)
    for win_ind, win in enumerate(list_wins):
        if win_ind in [0, 1]:
            print(f'sess_ind {sess_ind}, time window {win}ms')

            rate = list_rate_diffwin_trunc_all[sess_ind, win_ind].copy()
            # rate_sorted = rate.sort_index(axis=1)
            stm = rate.columns.copy()
            
            # Create a counting dictionary for each stimulus
            all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
            stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
            
            # convert to 3D response matrix
            min_num_trials = np.min(all_stm_counts)

            list_rate_tt = [None] * num_trial_types
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

            rate = np.stack(list_rate_tt, axis=2)
            rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials

            rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
            rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
                np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
            
            list_slopes_dr = list_slopes_diffwin_trunc_all_an_loglog[sess_ind, win_ind].copy()

            # trial order re-randomization
            for trial_type_ind in range(num_trial_types):
                rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, list_rand_trial_inds[trial_type_ind]]

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

                RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2))

                # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos))
                list_RSM[sampling_ind] = RSM.copy()
                
                count += 1
                if count % (n_sampling//2) == 0:
                    print(f'count: {count}')

            RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
            list_RSM_mean_asis = RSM_mean.copy()
            list_RSM_mean_asis2[win_ind] = list_RSM_mean_asis.copy()

            # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
            rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
            rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
            
            list_RSM_mean_RRneuron = np.zeros((len(list_target_slopes), num_trial_types, num_trial_types))
            for slope_ind, target_slope in enumerate(list_target_slopes):
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
                
                # # trial order re-randomization
                # for trial_type_ind in range(num_trial_types):
                #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

                list_rate_RRneuron_dr2[win_ind, slope_ind] = rate_RRneuron_dr.copy()

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
                        print(f'count: {count}')

                RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
                list_RSM_mean_RRneuron[slope_ind] = RSM_mean.copy()
            list_RSM_mean_RRneuron2[win_ind] = list_RSM_mean_RRneuron.copy()

    # Save into a file
    filename = 'RSM_diffwin_trunc_allneu_' + similarity_type + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis2', 'list_rate_RRneuron_dr2', 'list_RSM_mean_RRneuron2'], \
                     'list_RSM_mean_asis2': list_RSM_mean_asis2, 'list_rate_RRneuron_dr2': list_rate_RRneuron_dr2, 'list_RSM_mean_RRneuron2': list_RSM_mean_RRneuron2}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA within sessions
def RSA_withinsess_gratings(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 8
    n_neu_sampling = 10

    rng = np.random.default_rng(sess_ind) # match neuron partitioning
    # random.seed(0)

    # print(f'sess_ind {sess_ind}')

    # rate = list_rate_sg_all[sess_ind].copy()
    rate = list_rate_dg75_250_all[sess_ind].copy()

    # rate_sorted = rate.sort_index(axis=1)
    stm = rate.columns.copy()

    # Multiply by delta t to convert to spike counts
    rate = rate * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
    
    # convert to 3D response matrix
    min_num_trials = np.min(all_stm_counts)

    list_rate_tt = [None] * num_trial_types
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

    rate = np.stack(list_rate_tt, axis=2)
    rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials

    rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
    rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
        np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
    
    # list_slopes_dr = list_slopes_sg_all_an_loglog[sess_ind].copy()
    list_slopes_dr = list_slopes_dg75_250_all_an_loglog[sess_ind].copy()

    # trial order re-randomization
    for trial_type_ind in range(num_trial_types):
        rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

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
        #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

        list_rate_RRneuron_dr[slope_ind] = rate_RRneuron_dr.copy()

    # Iterate over neuron partitionings
    list_corr_withinsess_asis = np.full((n_neu_sampling, 3), np.nan)
    list_corr_withinsess2 = np.full((len(list_target_slopes), n_neu_sampling, 3), np.nan)    
    list_RSM_neu1_all = np.full((n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu2_all = np.full((n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu1_RRneuron_all = np.full((len(list_target_slopes), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu2_RRneuron_all = np.full((len(list_target_slopes), n_neu_sampling, num_trial_types, num_trial_types), np.nan)    
    for neu_sample_ind in range(n_neu_sampling):
        print(f'sess_ind {sess_ind}, neu_sample_ind = {neu_sample_ind}')
        
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
        list_RSM_neu1_all[neu_sample_ind], list_RSM_neu2_all[neu_sample_ind] = RSM_mean_neu1.copy(), RSM_mean_neu2.copy()

        # # exclude diagonal
        # RSM_mean_neu1[np.diag_indices(num_trial_types)] = np.nan
        # RSM_mean_neu2[np.diag_indices(num_trial_types)] = np.nan

        list_corr_withinsess_asis[neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
        bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
        list_corr_withinsess_asis[neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
        list_corr_withinsess_asis[neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
    
        for slope_ind, target_slope in enumerate(list_target_slopes):

            # RRneuron
            rate_RRneuron_dr = list_rate_RRneuron_dr[slope_ind].copy()

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
            list_RSM_neu1_RRneuron_all[slope_ind, neu_sample_ind], list_RSM_neu2_RRneuron_all[slope_ind, neu_sample_ind] = RSM_mean_neu1.copy(), RSM_mean_neu2.copy()

            # # exclude diagonal
            # RSM_mean_neu1[np.diag_indices(num_trial_types)] = np.nan
            # RSM_mean_neu2[np.diag_indices(num_trial_types)] = np.nan

            list_corr_withinsess2[slope_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_withinsess2[slope_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_withinsess2[slope_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_withinsess_sg_' + similarity_type + str(sess_ind) + '.pickle'
    filename = 'RSM_corr_withinsess_dg75_250_' + similarity_type + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:      
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis', 'list_corr_withinsess2', 'list_RSM_neu1_all', 'list_RSM_neu2_all', 'list_RSM_neu1_RRneuron_all', 'list_RSM_neu2_RRneuron_all'], \
                    'list_corr_withinsess_asis': list_corr_withinsess_asis, 'list_corr_withinsess2': list_corr_withinsess2, 'list_RSM_neu1_all': list_RSM_neu1_all, 'list_RSM_neu2_all': list_RSM_neu2_all,
                    'list_RSM_neu1_RRneuron_all': list_RSM_neu1_RRneuron_all, 'list_RSM_neu2_RRneuron_all': list_RSM_neu2_RRneuron_all}, f)
        
    print("Ended Process", c_proc.name)

# %%
# RSA within sessions (different time windows)
def RSA_withinsess_diffwin(sess_ind, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    n_neu_sampling = 10
    list_wins = np.arange(50, 250, 50) # time window size

    rng = np.random.default_rng(sess_ind) # match neuron partitioning
    # random.seed(0)

    # trial order re-randomization (to match trial order across time windows)
    rate = list_rate_diffwin_trunc_all[sess_ind, 0].copy()
    stm = rate.columns.copy()
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
    min_num_trials = np.min(all_stm_counts)   

    list_rand_trial_inds = np.zeros((num_trial_types, min_num_trials), dtype=int)
    for trial_type_ind in range(num_trial_types):
        list_rand_trial_inds[trial_type_ind] = rng.choice(range(min_num_trials), min_num_trials, replace=False)

    list_rate_sorted = np.empty(len(list_wins), dtype=object)
    list_rate_RRneuron_dr2 = np.empty((len(list_wins), len(list_target_slopes)), dtype=object)
    for win_ind, win in enumerate(list_wins):
        # print(f'sess_ind {sess_ind}, time window {win}ms')

        rate = list_rate_diffwin_trunc_all[sess_ind, win_ind].copy()
        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # convert to 3D response matrix
        min_num_trials = np.min(all_stm_counts)

        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        rate = np.stack(list_rate_tt, axis=2)
        rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
        list_rate_sorted[win_ind] = rate_sorted.copy()

        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
        
        list_slopes_dr = list_slopes_diffwin_trunc_all_an_loglog[sess_ind, win_ind].copy()

        # trial order re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, list_rand_trial_inds[trial_type_ind]]

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        for slope_ind, target_slope in enumerate(list_target_slopes):
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
            #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, rng.choice(range(min_num_trials), min_num_trials, replace=False)]

            list_rate_RRneuron_dr2[win_ind, slope_ind] = rate_RRneuron_dr.copy()

    # Iterate over neuron partitionings
    list_corr_withinsess_asis2 = np.full((len(list_wins), n_neu_sampling, 3), np.nan)
    list_corr_withinsess3 = np.full((len(list_wins), len(list_target_slopes), n_neu_sampling, 3), np.nan)    
    list_RSM_neu1_all2 = np.full((len(list_wins), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu2_all2 = np.full((len(list_wins), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu1_RRneuron_all2 = np.full((len(list_wins), len(list_target_slopes), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu2_RRneuron_all2 = np.full((len(list_wins), len(list_target_slopes), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    for neu_sample_ind in range(n_neu_sampling):
        # print(f'sess_ind {sess_ind}, neu_sample_ind = {neu_sample_ind}')

        rate_sorted = list_rate_sorted[0].copy()
        
        # Partition neurons
        neu_inds_permuted = rng.permutation(range(rate_sorted.shape[0]))
        neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 partitioning
        neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
        if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
            neu_div_inds2 = neu_div_inds2[:-1].copy()

        for win_ind, win in enumerate(list_wins):
            if win_ind in [0, 1]:
                print(f'sess_ind {sess_ind}, neu_sample_ind = {neu_sample_ind}, time window {win}ms')

                rate_sorted = list_rate_sorted[win_ind].copy()
                
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
                list_RSM_neu1_all2[win_ind, neu_sample_ind], list_RSM_neu2_all2[win_ind, neu_sample_ind] = RSM_mean_neu1.copy(), RSM_mean_neu2.copy()

                list_corr_withinsess_asis2[win_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_withinsess_asis2[win_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_withinsess_asis2[win_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
            
                for slope_ind, target_slope in enumerate(list_target_slopes):

                    # RRneuron
                    rate_RRneuron_dr = list_rate_RRneuron_dr2[win_ind, slope_ind].copy()

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
                    list_RSM_neu1_RRneuron_all2[win_ind, slope_ind, neu_sample_ind], list_RSM_neu2_RRneuron_all2[win_ind, slope_ind, neu_sample_ind] = RSM_mean_neu1.copy(), RSM_mean_neu2.copy()

                    list_corr_withinsess3[win_ind, slope_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                    bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                    list_corr_withinsess3[win_ind, slope_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                    list_corr_withinsess3[win_ind, slope_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_withinsess_diffwin_trunc_' + similarity_type + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:      
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis2', 'list_corr_withinsess3', 'list_RSM_neu1_all2', 'list_RSM_neu2_all2', 'list_RSM_neu1_RRneuron_all2', 'list_RSM_neu2_RRneuron_all2'], \
                    'list_corr_withinsess_asis2': list_corr_withinsess_asis2, 'list_corr_withinsess3': list_corr_withinsess3, 'list_RSM_neu1_all2': list_RSM_neu1_all2, 'list_RSM_neu2_all2': list_RSM_neu2_all2,
                    'list_RSM_neu1_RRneuron_all2': list_RSM_neu1_RRneuron_all2, 'list_RSM_neu2_RRneuron_all2': list_RSM_neu2_RRneuron_all2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# Effective dimensionality
def compute_eff_dim(sess_ind, n_trial_sampling=10):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 8
    rng = np.random.default_rng(sess_ind)

    list_dim_asis = np.zeros(num_trial_types)
    list_dim_RRneuron = np.zeros((len(list_target_slopes), num_trial_types))
    list_dim_global_asis = np.zeros(2)
    list_dim_global_RRneuron = np.zeros((len(list_target_slopes), 2))
    list_dim_sam_asis = np.zeros(n_trial_sampling)
    list_dim_sam_RRneuron = np.zeros((len(list_target_slopes), n_trial_sampling))

    print(f'sess_ind {sess_ind}')
    
    # rate = list_rate_sg_all[sess_ind].copy()
    rate = list_rate_dg75_250_all[sess_ind].copy()
    rate_sorted = rate.sort_index(axis=1)
    stm = rate.columns.copy()

    # Multiply by delta t to convert to spike counts
    rate_sorted = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    # list_slopes_dr = pd.DataFrame(list_slopes_sg_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()
    list_slopes_dr = pd.DataFrame(list_slopes_dg75_250_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()

    # pca
    n_components = rate_sorted.shape[0]
    pca = PCA(n_components=n_components)

    # Compute effective dimensionality for each stimulus
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        cf = np.cov(rate_sorted.loc[:, trial_type])
        list_dim_asis[trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_sorted.shape[0]
    cf_all = np.cov(rate_sorted)
    list_dim_global_asis[ 0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_sorted.shape[0]
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
    filename = 'eff_dim_DC_sg_' + str(sess_ind) + '.pickle'
    filename = 'eff_dim_DC_dg75_250_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_dim_asis', 'list_dim_RRneuron', 'list_dim_global_asis', 'list_dim_global_RRneuron', 'list_dim_sam_asis', 'list_dim_sam_RRneuron'],
                    'list_dim_asis': list_dim_asis, 'list_dim_RRneuron': list_dim_RRneuron, 'list_dim_global_asis': list_dim_global_asis, 'list_dim_global_RRneuron': list_dim_global_RRneuron,
                    'list_dim_sam_asis': list_dim_sam_asis, 'list_dim_sam_RRneuron': list_dim_sam_RRneuron}, f)

    print("Ended Process", c_proc.name)

# %%
# Effective dimensionality
def compute_eff_dim_diffwin(sess_ind, n_trial_sampling=100):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    list_wins = np.arange(50, 250, 50) # time window size

    num_trial_types = 119
    rng = np.random.default_rng(sess_ind)

    list_dim_asis2 = np.zeros((len(list_wins), num_trial_types))
    list_dim_RRneuron2 = np.zeros((len(list_wins), len(list_target_slopes), num_trial_types))
    list_dim_global_asis2 = np.zeros((len(list_wins), 2))
    list_dim_global_RRneuron2 = np.zeros((len(list_wins), len(list_target_slopes), 2))
    list_dim_sam_asis2 = np.zeros((len(list_wins), n_trial_sampling))
    list_dim_sam_RRneuron2 = np.zeros((len(list_wins), len(list_target_slopes), n_trial_sampling))

    for win_ind, win in enumerate(list_wins):
        if win_ind in [0, 1]:
            # print(f'sess_ind {sess_ind}, time window {win}ms')

            rate = list_rate_diffwin_trunc_all[sess_ind, win_ind].copy()
            rate_sorted = rate.sort_index(axis=1)
            stm = rate.columns.copy()

            # Create a counting dictionary for each stimulus
            all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
            stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

            # Compute mean & variance for each stimulus
            rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
            rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

            list_slopes_dr = pd.DataFrame(list_slopes_diffwin_trunc_all_an_loglog[sess_ind, win_ind], columns=rate_sorted_mean_coll.columns).copy()

            # pca
            n_components = rate_sorted.shape[0]
            pca = PCA(n_components=n_components)

            # Compute effective dimensionality for each stimulus
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                cf = np.cov(rate_sorted.loc[:, trial_type])
                list_dim_asis2[win_ind, trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_sorted.shape[0]
            cf_all = np.cov(rate_sorted)
            list_dim_global_asis2[win_ind, 0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_sorted.shape[0]
            cf_cen = np.cov(rate_sorted_mean_coll)
            list_dim_global_asis2[win_ind, 1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_sorted.shape[0]

            rand_tt_inds = rng.permutation(range(rate.shape[1]))
            rate = rate_sorted.iloc[:, rand_tt_inds].copy()
            for t_sam_ind in range(n_trial_sampling):
                rate_sam = np.full_like(rate_sorted_mean_coll, np.nan)
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    rate_tt = rate.loc[:, trial_type].copy()
                    rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
                cf_sam = np.cov(rate_sam)
                list_dim_sam_asis2[win_ind, t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_sorted.shape[0]

            # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
            rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
            rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

            for slope_ind, target_slope in enumerate(list_target_slopes):
                print(f'sess_ind {sess_ind}, time window {win}ms, target_slope = {target_slope:.1f}')

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
                    list_dim_RRneuron2[win_ind, slope_ind, trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_RRneuron_dr.shape[0]
                cf_all = np.cov(rate_RRneuron_dr)
                list_dim_global_RRneuron2[win_ind, slope_ind, 0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_RRneuron_dr.shape[0]
                cf_cen = np.cov(rate_mean_RRneuron_coll)
                list_dim_global_RRneuron2[win_ind, slope_ind, 1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_RRneuron_dr.shape[0]

                rate_RRneuron_dr = rate_RRneuron_dr.iloc[:, rand_tt_inds].copy()
                for t_sam_ind in range(n_trial_sampling):
                    rate_sam = np.full_like(rate_mean_RRneuron_coll, np.nan)
                    for trial_type_ind, trial_type in enumerate(all_stm_unique):
                        rate_tt = rate_RRneuron_dr.loc[:, trial_type].copy()
                        rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
                    cf_sam = np.cov(rate_sam)
                    list_dim_sam_RRneuron2[win_ind, slope_ind, t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_RRneuron_dr.shape[0]

                # # 2. Use eigenvalues
                # for trial_type_ind, trial_type in enumerate(all_stm_unique):
                #     rate_RRneuron_pca = pca.fit_transform(rate_RRneuron_dr.loc[:, trial_type].T).T            
                #     list_dim_RRneuron[slope_ind, trial_type_ind] = (np.sum(pca.explained_variance_))**2 / np.sum(pca.explained_variance_**2)

    # Save into a file
    filename = 'eff_dim_DC_diffwin_trunc_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_dim_asis2', 'list_dim_RRneuron2', 'list_dim_global_asis2', 'list_dim_global_RRneuron2', 'list_dim_sam_asis2', 'list_dim_sam_RRneuron2'],
                    'list_dim_asis2': list_dim_asis2, 'list_dim_RRneuron2': list_dim_RRneuron2, 'list_dim_global_asis2': list_dim_global_asis2, 'list_dim_global_RRneuron2': list_dim_global_RRneuron2,
                    'list_dim_sam_asis2': list_dim_sam_asis2, 'list_dim_sam_RRneuron2': list_dim_sam_RRneuron2}, f)

    print("Ended Process", c_proc.name)

# %%
# loading variables
with open('resp_matrix_ep_sg_all_32sess_gpu.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_sg_all = resp_matrix_ep_RS_all['list_rate_sg_all'].copy()
    list_slopes_sg_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_sg_all_an_loglog'].copy()
    list_sg_ori = resp_matrix_ep_RS_all['list_sg_ori'].copy()
    list_sg_sf = resp_matrix_ep_RS_all['list_sg_sf'].copy()
    list_sg_ph = resp_matrix_ep_RS_all['list_sg_ph'].copy()
    
with open('resp_matrix_ep_dg75_all_32sess_gpu.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_dg75_all = resp_matrix_ep_RS_all['list_rate_dg75_all'].copy()
    list_rate_dg75_250_all = resp_matrix_ep_RS_all['list_rate_dg75_250_all'].copy()
    list_slopes_dg75_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_dg75_all_an_loglog'].copy()
    list_slopes_dg75_250_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_dg75_250_all_an_loglog'].copy()
    list_dg75_ori = resp_matrix_ep_RS_all['list_dg75_ori'].copy()
    list_dg75_ct = resp_matrix_ep_RS_all['list_dg75_ct'].copy()
    list_sess_ids = resp_matrix_ep_RS_all['list_sess_ids'].copy()
    brain_observatory_sessid = resp_matrix_ep_RS_all['brain_observatory_sessid'].copy()

with open('resp_matrix_ep_diffwin_all_32sess_gpu.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_diffwin_trunc_all = resp_matrix_ep_RS_all['list_rate_diffwin_trunc_all'].copy()
    list_slopes_diffwin_trunc_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_diffwin_trunc_all_an_loglog'].copy()

# ABO Neuropixels
with open('resp_matrix_ep_RS_all_32sess_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_RS = resp_matrix_ep_RS_all['list_rate_RS'].copy()
    list_rate_RS_dr = resp_matrix_ep_RS_all['list_rate_RS_dr'].copy()
    list_rate_all = resp_matrix_ep_RS_all['list_rate_all'].copy()
    list_rate_all_dr = resp_matrix_ep_RS_all['list_rate_all_dr'].copy()
    list_slopes_RS_an_loglog = resp_matrix_ep_RS_all['list_slopes_RS_an_loglog'].copy()
    list_slopes_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_all_an_loglog'].copy()

# %%
# multiprocessing
num_sess = len(list_rate_sg_all) # static gratings (32 sessions)
num_sess = len(list_rate_dg75_250_all) # drifting gratings (58 sessions, only functional connectivity sessions are filled)

# decoding
decoder_type = 'SVM'
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, decoder_type] for sess_ind in range(num_sess)] # static gratings
        list_inputs = [[sess_ind, decoder_type] for sess_ind, sess_id in enumerate(list_sess_ids) if ~np.any(np.isin(sess_ind, [48, 49])) and ~np.isin(sess_id, brain_observatory_sessid)] # drifting gratings
        
        pool.starmap(decode_gratings, list_inputs)

# RSA
similarity_type = 'cos_sim'
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, similarity_type] for sess_ind in range(num_sess)] # static gratings
        list_inputs = [[sess_ind, similarity_type] for sess_ind, sess_id in enumerate(list_sess_ids) if ~np.any(np.isin(sess_ind, [48, 49])) and ~np.isin(sess_id, brain_observatory_sessid)] # drifting gratings
        
        pool.starmap(RSA_across_sesspairs_gratings, list_inputs)

# RSA
similarity_type = 'cos_sim'
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, similarity_type] for sess_ind in range(num_sess)] # static gratings
        list_inputs = [[sess_ind, similarity_type] for sess_ind, sess_id in enumerate(list_sess_ids) if ~np.any(np.isin(sess_ind, [48, 49])) and ~np.isin(sess_id, brain_observatory_sessid)] # drifting gratings
        
        pool.starmap(RSA_withinsess_gratings, list_inputs)

# effective dimensionality
n_trial_sampling = 100
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, n_trial_sampling] for sess_ind in range(num_sess)] # static gratings
        list_inputs = [[sess_ind, n_trial_sampling] for sess_ind, sess_id in enumerate(list_sess_ids) if ~np.any(np.isin(sess_ind, [48, 49])) and ~np.isin(sess_id, brain_observatory_sessid)] # drifting gratings
        
        pool.starmap(compute_eff_dim, list_inputs)
