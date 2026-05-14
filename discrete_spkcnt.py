# %%
# from pynwb import NWBHDF5IO
from scipy.io import savemat, loadmat
import mat73
import h5py
import hdf5storage as st
# from pymatreader import read_mat
import pickle
import os
import sys
import warnings
import multiprocessing as mp
import joblib
from joblib import Parallel, delayed
import contextlib

import numpy as np
import pandas as pd
import cupy as cp

from scipy.stats import wilcoxon, norm, kruskal, tukey_hsd, mode, spearmanr, rankdata, nbinom, poisson, binom, fit, multivariate_normal
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from scipy.optimize import minimize, curve_fit, minimize_scalar
from scipy.special import gammaln

import seaborn as sns
from copy import deepcopy as dc
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson, GeneralizedPoisson
from statsmodels.distributions.discrete import genpoisson_p
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from itertools import combinations, product, permutations, combinations_with_replacement
import math
import random
from time import time
import networkx as nx

from sklearn.svm import SVC
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
# negative log likelihood (modulated poisson)
def nll_shared_r_mp_test(r, counts_test, term_cons_test, mean_counts_test, all_stm_counts_test, mode='mean'):
    if r <= 0: return np.inf

    # full likelihood
    if mode == 'sum':
        term_rk = np.nansum(gammaln(counts_test + r) - gammaln(r)) - term_cons_test # term_cons_test has to be a sum
        term_r = (r * (np.log(r) - np.log(r + mean_counts_test))) * all_stm_counts_test
        term_r = np.sum(term_r[~np.isnan(term_r) & ~np.isinf(term_r)])
        term_k = (mean_counts_test * (np.log(mean_counts_test) - np.log(r + mean_counts_test))) * all_stm_counts_test
        term_k = np.sum(term_k[~np.isnan(term_k) & ~np.isinf(term_k)])
        log_likelihood = term_rk + term_r + term_k
    elif mode == 'mean':
        term_rk = gammaln(counts_test + r) - gammaln(r) - term_cons_test # term_cons_test has to be a vector
        term_r = np.repeat(r * (np.log(r) - np.log(r + mean_counts_test)), all_stm_counts_test)
        term_k = np.repeat(mean_counts_test * (np.log(mean_counts_test) - np.log(r + mean_counts_test)), all_stm_counts_test)
        log_likelihood = term_rk + term_r + term_k
        log_likelihood = np.mean(log_likelihood[~np.isnan(log_likelihood) & ~np.isinf(log_likelihood)])

    return -log_likelihood

# negative log likelihood (FF, LLL)
def nll_nbp_test(params, mu_neu_test, all_stm_counts_test, k_test, log_fact_counts_test, model='LLL', mode='mean'):
    if model == 'LLL':
        # LLL
        a, b = params
        # Predicted Variance: v = 10^b * mu^a
        log_v_pred = a * np.log10(mu_neu_test) + b
        v_pred = pow(10, log_v_pred)
    elif model == 'FF':
        # FF
        ff = params
        # Predicted Variance: v = FF * mu
        v_pred = ff * mu_neu_test
    else:
        raise Exception('Model not supported')

    # divide stimuli based on predicted variance, not sample variance
    mask_nb = v_pred > mu_neu_test
    mask_poiss = ~mask_nb
    
    # Calculate r and p of NB
    r = (mu_neu_test**2) / (v_pred - mu_neu_test)
    log_p = np.log(r) - np.log(r + mu_neu_test)
    log_1_minus_p = np.log(mu_neu_test) - np.log(r + mu_neu_test) # 1-p = mu/(r+mu) because p = r/(r+mu)

    ll_trials = np.zeros(np.sum(all_stm_counts_test))
    ll_trials[mask_nb] = (gammaln(k_test + r) - log_fact_counts_test - gammaln(r) +
                        r * log_p + k_test * log_1_minus_p)[mask_nb] # NB
    ll_trials[mask_poiss] = (k_test * np.log(mu_neu_test) - mu_neu_test - log_fact_counts_test)[mask_poiss] # Poisson
    if mode == 'sum':
        nll = -np.nansum(ll_trials)
    elif mode == 'mean':
        nll = -np.nanmean(ll_trials)
    
    return nll

# %%
# suppress all warnings
@contextlib.contextmanager
def suppress_all_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        old_np = np.seterr(all='ignore')
        old_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        try:
            yield
        finally:
            sys.stderr.close()
            sys.stderr = old_stderr
            np.seterr(**old_np)

# %%
def fit_discrete_models(sess_ind, mode='mean'):

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    # ignore warnings
    with suppress_all_warnings():
        print(f'session index: {sess_ind}')
        
        num_trial_types = 119
        num_trial_types_sg = 121
        num_trials = 50
        num_trials_sg = 197 # max number of trials for static gratings grayscreen
        n_splits = 10 # cross-validation number of folds

        # natural scenes
        rate = dc(list_rate_all[sess_ind])
        stm = rate.columns.copy()

        # Multiply by delta t to convert to spike counts
        rate = rate * 0.25

        # Create a counting dictionary for each stimulus
        rate_sorted = rate.sort_index(axis=1)
        stm_sorted = np.array(sorted(stm))
        num_neurons = rate_sorted.shape[0]
        labels = rate_sorted.columns

        all_stm_unique, all_stm_counts = np.unique(stm_sorted, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

        rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
        rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)
        
        counts_3d = np.full((num_neurons, num_trial_types, num_trials), np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            num_trials_temp = rate_sorted.loc[:, trial_type].shape[1]
            counts_3d[:, trial_type_ind, :num_trials_temp] = rate_sorted.loc[:, trial_type].copy()
        log_fact_counts = gammaln(counts_3d + 1)

        # static gratings
        rate_sg = dc(list_rate_sg_all[sess_ind])
        stm_sg = rate_sg.columns.copy()

        rate_sg = rate_sg * 0.25

        rate_sorted_sg = rate_sg.sort_index(axis=1)
        stm_sorted_sg = np.array(sorted(stm_sg))
        num_neurons = rate_sorted_sg.shape[0]
        labels_sg = rate_sorted_sg.columns

        all_stm_unique_sg, all_stm_counts_sg = np.unique(stm_sorted_sg, return_counts=True)
        stm_cnt_dict_sg = dict(zip(all_stm_unique_sg, all_stm_counts_sg))

        rate_sorted_mean_coll_sg, rate_sorted_var_coll_sg = compute_mean_var_trial_collapse(stm_cnt_dict_sg, rate_sorted_sg)

        counts_3d_sg = np.full((num_neurons, num_trial_types_sg, num_trials_sg), np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique_sg):
            num_trials_temp = rate_sorted_sg.loc[:, trial_type].shape[1]
            counts_3d_sg[:, trial_type_ind, :num_trials_temp] = rate_sorted_sg.loc[:, trial_type].copy()
        log_fact_counts_sg = gammaln(counts_3d_sg + 1)
        
        # fit parameters per neuron using MLE (maximum likelihood estimation)
        
        list_r_estim_neu = np.full(num_neurons, np.nan)
        list_p_estim_neu = np.full(num_neurons, np.nan)
        list_lam_estim_neu = np.full(num_neurons, np.nan)
        list_n_estim_neu = np.full(num_neurons, np.nan)
        list_r_estim_neu = np.full((num_neurons, num_trial_types), np.nan)
        list_p_estim_neu = np.full((num_neurons, num_trial_types), np.nan)
        list_lam_estim_neu = np.full((num_neurons, num_trial_types), np.nan)
        list_n_estim_neu = np.full((num_neurons, num_trial_types), np.nan)

        list_rp_estim_neu = np.full((num_neurons, num_trial_types+1), np.nan)

        list_lll_estim_neu = np.full((num_neurons, 2), np.nan)
        list_ff_estim_neu = np.full(num_neurons, np.nan)
        
        list_nll_neu = np.full(num_neurons, np.nan) # negative log likelihood
        list_nll_neu_temp = np.full((num_neurons, num_trial_types), np.nan) # negative log likelihood
        list_aic_neu = np.full(num_neurons, np.nan)
        list_rmse_neu = np.full(num_neurons, np.nan) # RMSE for var (mean is already matched)
        
        list_nll_neu_cv = np.full((num_neurons, n_splits), np.nan) # negative log likelihood
        list_aic_neu_cv = np.full((num_neurons, n_splits), np.nan)
        list_rmse_neu_cv = np.full((num_neurons, n_splits), np.nan) # RMSE for var (mean is already matched)
        
        list_nll_neu_cv_ns2sg = np.full((num_neurons, n_splits), np.nan) # negative log likelihood
        list_aic_neu_cv_ns2sg = np.full((num_neurons, n_splits), np.nan)
        list_rmse_neu_cv_ns2sg = np.full((num_neurons, n_splits), np.nan) # RMSE for var (mean is already matched)

        list_nll_neu_cv_sg = np.full((num_neurons, n_splits), np.nan) # negative log likelihood
        list_aic_neu_cv_sg = np.full((num_neurons, n_splits), np.nan)
        list_rmse_neu_cv_sg = np.full((num_neurons, n_splits), np.nan) # RMSE for var (mean is already matched)    

        start = time()
        for neu_ind in rate_sorted.index:
            # 1. fit for each neuron-stimulus combination (NB+P)
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                if rate_sorted_mean_coll.loc[neu_ind, trial_type] > 0:
                    nbmodel = NegativeBinomial(rate_sorted.loc[neu_ind, trial_type], np.ones_like(rate_sorted.loc[neu_ind, trial_type])) # MLE (default: bfgs)
                    if rate_sorted_mean_coll.loc[neu_ind, trial_type] >= rate_sorted_var_coll.loc[neu_ind, trial_type]: # poisson
                        r_estim, p_estim = np.inf, 1
                    else: # negative binomial
                        res = nbmodel.fit(method='bfgs', disp=False) # convergence message off
                        mu_estim, var_G_estim = res.params
                        mu_estim = np.exp(mu_estim) # caution
                        r_estim = 1/var_G_estim
                        p_estim = r_estim / (r_estim + mu_estim)
                else:
                    r_estim, p_estim = np.nan, np.nan
                list_r_estim_neu[neu_ind, trial_type_ind], list_p_estim_neu[neu_ind, trial_type_ind] = r_estim, p_estim

            # 2. fit one r and num_trial_types p's simultaneously (MP)
            if np.mean(rate_sorted.loc[neu_ind, :]) > 0: # all neurons in all sessions
                mean_neu, var_neu = np.mean(rate_sorted.loc[neu_ind, :]), np.var(rate_sorted.loc[neu_ind, :], ddof=1)
                if var_neu > mean_neu:
                    # natural scenes
                    counts = rate_sorted.loc[neu_ind, :].values
                    mean_counts = rate_sorted_mean_coll.loc[neu_ind, :].values
                    sum_counts = mean_counts * all_stm_counts
                    num_tot_trials = rate_sorted.shape[1]
                    term_cons = np.nansum(log_fact_counts[neu_ind])

                    # static gratings
                    counts_sg = rate_sorted_sg.loc[neu_ind, :].values
                    mean_counts_sg = rate_sorted_mean_coll_sg.loc[neu_ind, :].values
                    term_cons_sg = np.nansum(log_fact_counts_sg[neu_ind])

                    # negative log likelihood function
                    def nll_shared_r(r):
                        if r <= 0: return np.inf
                        # full likelihood
                        term_rk = np.nansum(gammaln(counts + r) - gammaln(r)) - term_cons
                        term_r = (r * (np.log(r) - np.log(r + mean_counts))) * all_stm_counts
                        term_r = np.sum(term_r[~np.isnan(term_r) & ~np.isinf(term_r)])
                        term_k = (mean_counts * (np.log(mean_counts) - np.log(r + mean_counts))) * all_stm_counts
                        term_k = np.sum(term_k[~np.isnan(term_k) & ~np.isinf(term_k)])
                        log_likelihood = term_rk + term_r + term_k

                        return -log_likelihood

                    res = minimize_scalar(nll_shared_r, bounds=(1e-3, 1e+3), method='bounded')
                    # res = minimize(nll_shared_r, x0=1, bounds=[(1e-3, 1e+3)], method='trust-constr')
                    r_estim_neu = res.x
                    list_rp_estim_neu[neu_ind, 0] = r_estim_neu
                    list_rp_estim_neu[neu_ind, 1:] = r_estim_neu / (r_estim_neu + mean_counts)
                    nll = res.fun
                    list_nll_neu[neu_ind] = nll

                    num_params = 1 + num_trial_types
                    v_pred = (1/r_estim_neu) * mean_counts**2 + mean_counts
                    var_counts = rate_sorted_var_coll.loc[neu_ind, :].values

                    list_aic_neu[neu_ind] = 2 * (num_params + nll)
                    list_rmse_neu[neu_ind] = np.sqrt(np.mean((v_pred - var_counts)**2))

                    # cross-validation for comparison with other models
                    stkfold = StratifiedKFold(n_splits=n_splits)
                    for split_ind, (train_index, test_index) in enumerate(stkfold.split(counts[:, np.newaxis], labels)):
                        # natural scenes
                        counts_train, counts_test = counts[train_index], counts[test_index]
                        label_train, label_test = labels[train_index], labels[test_index]
                        
                        mean_counts_train = pd.Series(counts_train, index=label_train).groupby(level=0).mean().values # stimuli sorted
                        log_fact_counts_train = log_fact_counts[neu_ind][~np.isnan(log_fact_counts[neu_ind])] # flattened
                        log_fact_counts_train = log_fact_counts_train[train_index]
                        term_cons_train = np.sum(log_fact_counts_train)
                        _, all_stm_counts_train = np.unique(label_train, return_counts=True)

                        mean_counts_test = pd.Series(counts_test, index=label_test).groupby(level=0).mean().values # stimuli sorted
                        log_fact_counts_test = log_fact_counts[neu_ind][~np.isnan(log_fact_counts[neu_ind])] # flattened
                        log_fact_counts_test = log_fact_counts_test[test_index]
                        if mode == 'sum':
                            term_cons_test = np.sum(log_fact_counts_test) # sum nll
                        elif mode == 'mean':
                            term_cons_test = log_fact_counts_test # mean nll
                        _, all_stm_counts_test = np.unique(label_test, return_counts=True)

                        # static gratings
                        counts_test_sg = counts_sg
                        label_test_sg = labels_sg
                        
                        mean_counts_test_sg = pd.Series(counts_test_sg, index=label_test_sg).groupby(level=0).mean().values # stimuli sorted
                        log_fact_counts_test_sg = log_fact_counts_sg[neu_ind][~np.isnan(log_fact_counts_sg[neu_ind])] # flattened
                        if mode == 'sum':
                            term_cons_test_sg = np.sum(log_fact_counts_test_sg) # sum nll
                        elif mode == 'mean':
                            term_cons_test_sg = log_fact_counts_test_sg # mean nll
                        _, all_stm_counts_test_sg = np.unique(label_test_sg, return_counts=True)

                        # negative log likelihood function
                        def nll_shared_r_train(r):
                            if r <= 0: return np.inf

                            # full likelihood
                            term_rk = np.nansum(gammaln(counts_train + r) - gammaln(r)) - term_cons_train
                            term_r = (r * (np.log(r) - np.log(r + mean_counts_train))) * all_stm_counts_train
                            term_r = np.sum(term_r[~np.isnan(term_r) & ~np.isinf(term_r)])
                            term_k = (mean_counts_train * (np.log(mean_counts_train) - np.log(r + mean_counts_train))) * all_stm_counts_train
                            term_k = np.sum(term_k[~np.isnan(term_k) & ~np.isinf(term_k)])
                            log_likelihood = term_rk + term_r + term_k

                            return -log_likelihood
                    
                        res = minimize_scalar(nll_shared_r_train, bounds=(1e-3, 1e+3), method='bounded')
                        r_estim_neu = res.x
                        
                        # calculate metrics for test data
                        # natural scenes to natural scenes
                        nll = nll_shared_r_mp_test(r_estim_neu, counts_test, term_cons_test, mean_counts_test, all_stm_counts_test, mode=mode)
                        list_nll_neu_cv[neu_ind, split_ind] = nll

                        num_params = 1 + num_trial_types
                        v_pred_test = (1/r_estim_neu) * mean_counts_test**2 + mean_counts_test
                        var_counts_test = pd.Series(counts_test, index=label_test).groupby(level=0).var().values # stimuli sorted

                        list_aic_neu_cv[neu_ind, split_ind] = 2 * (num_params + nll)
                        list_rmse_neu_cv[neu_ind, split_ind] = np.sqrt(np.mean((v_pred_test - var_counts_test)**2))

                        # natural scenes to static gratings
                        nll_sg = nll_shared_r_mp_test(r_estim_neu, counts_test_sg, term_cons_test_sg, mean_counts_test_sg, all_stm_counts_test_sg, mode=mode)
                        list_nll_neu_cv_ns2sg[neu_ind, split_ind] = nll_sg

                        num_params_sg = num_trial_types_sg # exclude parameters estimated in natural scenes from degree of freedom 
                        v_pred_test_sg = (1/r_estim_neu) * mean_counts_test_sg**2 + mean_counts_test_sg
                        var_counts_test_sg = pd.Series(counts_test_sg, index=label_test_sg).groupby(level=0).var().values # stimuli sorted

                        list_aic_neu_cv_ns2sg[neu_ind, split_ind] = 2 * (num_params_sg + nll_sg)
                        list_rmse_neu_cv_ns2sg[neu_ind, split_ind] = np.sqrt(np.mean((v_pred_test_sg - var_counts_test_sg)**2))

                else:
                    list_rp_estim_neu[neu_ind] = np.concatenate([[np.inf], np.ones(num_trial_types)])

            # 3. LLL (loglog linear)
            if np.mean(rate_sorted.loc[neu_ind, :]) > 0:
                # natural scenes
                counts = counts_3d[neu_ind] # (num_trial_types, num_trials)
                mean_neu = rate_sorted_mean_coll.loc[neu_ind].values # (num_trial_types,)
                var_neu = rate_sorted_var_coll.loc[neu_ind].values # (num_trial_types,)

                k = counts.flatten()[~np.isnan(counts.flatten())]
                mu_neu = np.repeat(mean_neu, all_stm_counts)
                log_fact_counts_neu = log_fact_counts[neu_ind]
                log_fact_counts_neu = log_fact_counts_neu.flatten()[~np.isnan(log_fact_counts_neu.flatten())]

                # static gratings
                counts_sg = counts_3d_sg[neu_ind] # (num_trial_types, num_trials)
                mean_neu_sg = rate_sorted_mean_coll_sg.loc[neu_ind].values # (num_trial_types,)
                var_neu_sg = rate_sorted_var_coll_sg.loc[neu_ind].values # (num_trial_types,)

                k_sg = counts_sg.flatten()[~np.isnan(counts_sg.flatten())]
                mu_neu_sg = np.repeat(mean_neu_sg, all_stm_counts_sg)
                log_fact_counts_neu_sg = log_fact_counts_sg[neu_ind]
                log_fact_counts_neu_sg = log_fact_counts_neu_sg.flatten()[~np.isnan(log_fact_counts_neu_sg.flatten())]

                # Objective Function (Negative Log Likelihood)
                def nll_nbp(params):
                    # LLL
                    a, b = params
                    # Predicted Variance: v = 10^b * mu^a
                    log_v_pred = a * np.log10(mu_neu) + b
                    v_pred = pow(10, log_v_pred)

                    # divide stimuli based on predicted variance, not sample variance
                    mask_nb = v_pred > mu_neu
                    mask_poiss = ~mask_nb
                    
                    # Calculate r and p of NB
                    r = (mu_neu**2) / (v_pred - mu_neu)
                    log_p = np.log(r) - np.log(r + mu_neu)
                    log_1_minus_p = np.log(mu_neu) - np.log(r + mu_neu) # 1-p = mu/(r+mu) because p = r/(r+mu)

                    nll_trials = np.zeros(np.sum(all_stm_counts))
                    nll_trials[mask_nb] = (gammaln(k + r) - log_fact_counts_neu - gammaln(r) +
                                           r * log_p + k * log_1_minus_p)[mask_nb] # NB
                    nll_trials[mask_poiss] = (k * np.log(mu_neu) - mu_neu - log_fact_counts_neu)[mask_poiss] # Poisson
                    
                    return -np.nansum(nll_trials)
                
                # Initial guess: Simple linear regression on log-log of sample stats
                bool_mean_notzero = mean_neu > 0
                p_init = np.polyfit(np.log10(mean_neu[bool_mean_notzero]).astype(np.float32),
                                    np.log10(var_neu[bool_mean_notzero]).astype(np.float32), 1) # LLL
                res = minimize(nll_nbp, p_init, method='L-BFGS-B')
                nll = res.fun
                list_nll_neu[neu_ind] = nll
                
                # LLL
                a_estim = res.x[0]
                b_estim = res.x[1]
                list_lll_estim_neu[neu_ind] = [a_estim, b_estim]
                num_params = 2 + num_trial_types
                v_pred = pow(10, a_estim * np.log10(mean_neu) + b_estim)

                list_aic_neu[neu_ind] = 2 * (num_params + nll)
                list_rmse_neu[neu_ind] = np.sqrt(np.nanmean((v_pred - var_neu)**2))

                # cross-validation for comparison with other models
                stkfold = StratifiedKFold(n_splits=n_splits)              
                for split_ind, (train_index, test_index) in enumerate(stkfold.split(k[:, np.newaxis], labels)):
                    # natural scenes
                    k_train, k_test = k[train_index], k[test_index]
                    label_train, label_test = labels[train_index], labels[test_index]
                    
                    mean_neu_train = pd.Series(k_train, index=label_train).groupby(level=0).mean().values # stimuli sorted
                    var_neu_train = pd.Series(k_train, index=label_train).groupby(level=0).var().values
                    _, all_stm_counts_train = np.unique(label_train, return_counts=True)
                    mu_neu_train = np.repeat(mean_neu_train, all_stm_counts_train)
                    log_fact_counts_train = log_fact_counts[neu_ind][~np.isnan(log_fact_counts[neu_ind])]
                    log_fact_counts_train = log_fact_counts_train[train_index]

                    mean_neu_test = pd.Series(k_test, index=label_test).groupby(level=0).mean().values # stimuli sorted
                    var_neu_test = pd.Series(k_test, index=label_test).groupby(level=0).var().values
                    _, all_stm_counts_test = np.unique(label_test, return_counts=True)
                    mu_neu_test = np.repeat(mean_neu_test, all_stm_counts_test)
                    log_fact_counts_test = log_fact_counts[neu_ind][~np.isnan(log_fact_counts[neu_ind])]
                    log_fact_counts_test = log_fact_counts_test[test_index]

                    # static gratings
                    k_test_sg = k_sg
                    label_test_sg = labels_sg

                    mean_neu_test_sg = pd.Series(k_test_sg, index=label_test_sg).groupby(level=0).mean().values # stimuli sorted
                    var_neu_test_sg = pd.Series(k_test_sg, index=label_test_sg).groupby(level=0).var().values
                    _, all_stm_counts_test_sg = np.unique(label_test_sg, return_counts=True)
                    mu_neu_test_sg = np.repeat(mean_neu_test_sg, all_stm_counts_test_sg)
                    log_fact_counts_test_sg = log_fact_counts_sg[neu_ind][~np.isnan(log_fact_counts_sg[neu_ind])]

                    # Objective Function (Negative Log Likelihood)
                    def nll_nbp_train(params):
                        # LLL
                        a, b = params
                        # Predicted Variance: v = 10^b * mu^a
                        log_v_pred = a * np.log10(mu_neu_train) + b
                        v_pred = pow(10, log_v_pred)

                        # divide stimuli based on predicted variance, not sample variance
                        mask_nb = v_pred > mu_neu_train
                        mask_poiss = ~mask_nb
                        
                        # Calculate r and p of NB
                        r = (mu_neu_train**2) / (v_pred - mu_neu_train)
                        log_p = np.log(r) - np.log(r + mu_neu_train)
                        log_1_minus_p = np.log(mu_neu_train) - np.log(r + mu_neu_train) # 1-p = mu/(r+mu) because p = r/(r+mu)

                        nll_trials = np.zeros(np.sum(all_stm_counts_train))
                        nll_trials[mask_nb] = (gammaln(k_train + r) - log_fact_counts_train - gammaln(r) +
                                            r * log_p + k_train * log_1_minus_p)[mask_nb] # NB
                        nll_trials[mask_poiss] = (k_train * np.log(mu_neu_train) - mu_neu_train - log_fact_counts_train)[mask_poiss] # Poisson
                        
                        return -np.nansum(nll_trials)
                    
                    # Initial guess: Simple linear regression on log-log of sample stats
                    bool_mean_notzero = mean_neu_train > 0
                    p_init = np.polyfit(np.log10(mean_neu_train[bool_mean_notzero]).astype(np.float32),
                                        np.log10(var_neu_train[bool_mean_notzero]).astype(np.float32), 1) # LLL
                    res = minimize(nll_nbp_train, p_init, method='L-BFGS-B')

                    # calculate metrics for test data
                    # LLL
                    # natural scenes
                    a_estim = res.x[0]
                    b_estim = res.x[1]
                    nll = nll_nbp_test((a_estim, b_estim), mu_neu_test, all_stm_counts_test, k_test, log_fact_counts_test, model='LLL', mode=mode)
                    list_nll_neu_cv[neu_ind, split_ind] = nll
                    num_params = 2 + num_trial_types
                    v_pred_test = pow(10, a_estim * np.log10(mean_neu_test) + b_estim)

                    # static gratings
                    a_estim = res.x[0]
                    b_estim = res.x[1]
                    nll_sg = nll_nbp_test((a_estim, b_estim), mu_neu_test_sg, all_stm_counts_test_sg, k_test_sg, log_fact_counts_test_sg, model='LLL', mode=mode)
                    list_nll_neu_cv_ns2sg[neu_ind, split_ind] = nll_sg
                    num_params_sg = num_trial_types_sg # exclude parameters estimated in natural scenes from degree of freedom 
                    v_pred_test_sg = pow(10, a_estim * np.log10(mean_neu_test_sg) + b_estim)

                    list_aic_neu_cv[neu_ind, split_ind] = 2 * (num_params + nll)
                    list_rmse_neu_cv[neu_ind, split_ind] = np.sqrt(np.nanmean((v_pred_test - var_neu_test)**2))
                    list_aic_neu_cv_ns2sg[neu_ind, split_ind] = 2 * (num_params_sg + nll_sg)
                    list_rmse_neu_cv_ns2sg[neu_ind, split_ind] = np.sqrt(np.nanmean((v_pred_test_sg - var_neu_test_sg)**2))

            if neu_ind % 5 == 0:
                print(f'sess_ind: {sess_ind}, neu_ind: {neu_ind}/{rate_sorted.shape[0]}, duration: {(time() - start)/60:.2f} min')
                start = time()

    # Save into a file
    filename = 'poisson_fit_rp_sep_ABO_' + str(sess_ind) + '.pickle'
    filename = 'poisson_fit_rp_semipool_ABO_' + str(sess_ind) + '.pickle'
    filename = 'poisson_fit_lll_nbp_ABO_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        # pickle.dump({'tree_variables': ['list_r_estim_neu', 'list_p_estim_neu'],
        #              'list_r_estim_neu': list_r_estim_neu, 'list_p_estim_neu': list_p_estim_neu}, f)
        
        # pickle.dump({'tree_variables': ['list_rp_estim_neu', 'list_nll_neu', 'list_aic_neu', 'list_rmse_neu', 'list_nll_neu_cv', 'list_aic_neu_cv', 'list_rmse_neu_cv',
        #                                 'list_nll_neu_cv_ns2sg', 'list_aic_neu_cv_ns2sg', 'list_rmse_neu_cv_ns2sg'],
        #              'list_rp_estim_neu': list_rp_estim_neu, 'list_nll_neu': list_nll_neu, 'list_aic_neu': list_aic_neu, 'list_rmse_neu': list_rmse_neu,
        #              'list_nll_neu_cv': list_nll_neu_cv, 'list_aic_neu_cv': list_aic_neu_cv, 'list_rmse_neu_cv': list_rmse_neu_cv,
        #              'list_nll_neu_cv_ns2sg': list_nll_neu_cv_ns2sg, 'list_aic_neu_cv_ns2sg': list_aic_neu_cv_ns2sg, 'list_rmse_neu_cv_ns2sg': list_rmse_neu_cv_ns2sg}, f)
        pickle.dump({'tree_variables': ['list_lll_estim_neu', 'list_nll_neu', 'list_aic_neu', 'list_rmse_neu', 'list_nll_neu_cv', 'list_aic_neu_cv', 'list_rmse_neu_cv',
                                        'list_nll_neu_cv_ns2sg', 'list_aic_neu_cv_ns2sg', 'list_rmse_neu_cv_ns2sg'],
                     'list_lll_estim_neu': list_lll_estim_neu, 'list_nll_neu': list_nll_neu, 'list_aic_neu': list_aic_neu, 'list_rmse_neu': list_rmse_neu,
                     'list_nll_neu_cv': list_nll_neu_cv, 'list_aic_neu_cv': list_aic_neu_cv, 'list_rmse_neu_cv': list_rmse_neu_cv,
                     'list_nll_neu_cv_ns2sg': list_nll_neu_cv_ns2sg, 'list_aic_neu_cv_ns2sg': list_aic_neu_cv_ns2sg, 'list_rmse_neu_cv_ns2sg': list_rmse_neu_cv_ns2sg}, f)
        
    print("Ended Process",c_proc.name)

# %%
def _nearest_spd(A, eps=1e-8):
    """Project a symmetric matrix to the nearest SPD (symmetric positive definite) by eigenvalue clipping."""
    A = (A + A.T) / 2.0
    A[np.isnan(A)] = 0 # correlation NaN is from all-zero neurons; convert it to 0
    A[np.diag_indices(A.shape[0])] = 1 # diagonal must be 1
    w, v = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    if not np.allclose(A, (v * w) @ v.T):
        print(f'A is different from reconstructed A')
    return (v * w) @ v.T

def _nb_ppf(u, r, p):
    # scipy nbinom is discrete; use ppf safely in (0,1)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return nbinom.ppf(u, n=r, p=p).astype(int)

def _poiss_ppf(u, mu):
    # scipy poisson is discrete; use ppf safely in (0,1)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return poisson.ppf(u, mu=mu).astype(int)

def _bin_ppf(u, n, p):
    # scipy poisson is discrete; use ppf safely in (0,1)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return binom.ppf(u, n, p).astype(int)

def sample_nb_with_copula(spkcnt_target, marginal_params, n_samples=50, seed=0, model='negbinom'):
    """
    spkcnt_target: np.ndarray of shape (n_features, n_obs) with nonnegative integers
    marginal_params: array of shape (n_features, 2) with fitted r and p of negative binomial distribution
    Returns: samples of shape (n_features, n_samples)
    """
    rng = np.random.default_rng(seed)
    n_features, n_obs = spkcnt_target.shape

    # Correlation across features (variables), using observations as columns
    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        R = np.corrcoef(spkcnt_target, rowvar=True)
    R = _nearest_spd(R, eps=1e-8)

    # Sample from Gaussian copula
    # Cholesky (fallback to eig if needed)
    try:
        L = np.linalg.cholesky(R) # n_features x n_features, lower triangular matrix
    except np.linalg.LinAlgError:
        # Robust fallback
        w, v = np.linalg.eigh(R)
        L = v @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))

    Z_samp = rng.standard_normal(size=(n_features, n_samples))
    Z_corr = L @ Z_samp # target correlation imposed
    U_samp = norm.cdf(Z_corr.T).T # uniforms in (0,1) (n_features x n_samples)

    # Apply inverse CDF of each feature's marginal
    out = np.empty((n_features, n_samples), dtype=int)
    if model == 'negbinom':
        for j, (r, p) in enumerate(marginal_params):
            mu = np.array(spkcnt_target)[j].mean()
            var_ = np.array(spkcnt_target)[j].var(ddof=1)

            # use optimal r for each neuron-stimulus combination
            if np.isnan(r):
                out[j] = np.zeros_like(U_samp[j])
            else: 
                if np.isinf(r): # var <= mean
                    out[j] = _poiss_ppf(U_samp[j], mu)
                else:
                    if mu == 0: # mean = 0
                        out[j] = np.zeros_like(U_samp[j])
                    else: # var > mean
                        out[j] = _nb_ppf(U_samp[j], r, p)

    elif model == 'poisson':
        for j, lam in enumerate(marginal_params):
            mu = np.array(spkcnt_target)[j].mean()
            if mu == 0: # mean = 0
                out[j] = np.zeros_like(U_samp[j])
            else: # mean > 0
                out[j] = _poiss_ppf(U_samp[j], lam)

    return out

def sample_nb_indep(spkcnt_target, marginal_params, sess_ind, trial_type_ind, n_samples=50, model='katz', margin=0.05):
    """
    spkcnt_target: np.ndarray of shape (n_features, n_obs) with nonnegative integers
    marginal_params: array of shape (n_features, 2) with fitted r and p of negative binomial distribution
    Returns: samples of shape (n_features, n_samples)
    """
    n_features, n_obs = spkcnt_target.shape

    # Apply inverse CDF of each feature's marginal
    out = np.empty((n_features, n_samples), dtype=int)
    if model == 'katz':
        for j, (r, p, lam, n) in enumerate(marginal_params):
            mu = np.array(spkcnt_target)[j].mean()
            var_ = np.array(spkcnt_target)[j].var(ddof=1)
            random_state = np.abs(hash((sess_ind, j))) % (2**32) # unique interger for each combination, needs bounding to support scipy random_state

            if mu == 0: # mean = 0
                out[j] = np.zeros(n_samples)
            else: # mean > 0
                if var_ > mu*(1+margin):
                    out[j] = nbinom.rvs(n=r, p=p, size=n_samples, random_state=random_state)
                elif (var_ >= mu*(1-margin)) & (var_ < mu*(1+margin)):
                    out[j] = poisson.rvs(lam, size=n_samples, random_state=random_state)
                else:
                    try:
                        n = int(np.round(n))
                    except:
                        print(f'neuron {j}, n = {n}, mu {mu}, var_ {var_}')
                    if n == 0:
                        n = 1; p = mu / n # set n >= 1, match mean
                    out[j] = binom.rvs(n, p, size=n_samples, random_state=random_state)

    return out

# %%
# sample spike count from discrete probability distributions (use Gaussian copula)
def sample_spkcnt_copula(sess_ind, method, n_samples, model='negbinom'):

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        list_rescale_factors = [10, 1, 0]
        shift_factor = 0
        if method == 'resc_r':
            list_factors = dc(list_rescale_factors)

        print(f'session index: {sess_ind}')
        
        rate_sorted = list_rate_all[sess_ind].sort_index(axis=1)
        stm = rate_sorted.columns.copy()
        num_neurons = rate_sorted.shape[0]

        # Multiply by delta t to convert to spike counts
        rate_sorted = rate_sorted * 0.25
        # print(rate_sorted.shape[0])

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

        # Compute mean & variance for each stimulus
        rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
        rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

        list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()

        if model == 'negbinom':
            list_rate_nb = np.full((len(list_factors), rate_sorted.shape[0], len(all_stm_unique)*n_samples), np.nan)
            for f_ind, factor in enumerate(list_factors):
                if f_ind == 1:
                    start_time = time()

                    if method == 'resc_r':
                        # r rescaling
                        r_estim_neu = list_r_estim_neu2[sess_ind].copy()
                        rescaled_r = pow(10, factor * np.log10(r_estim_neu) + shift_factor)
                        if rescaled_r.ndim == 1: # if rescaled_r's shape is (num_neurons,)
                            rescaled_r = rescaled_r[:, np.newaxis]
                            use_rrep = True
                        else:
                            use_rrep = False
                        new_p = list_p_estim_neu2[sess_ind].copy() # num_neurons x num_trial_types
                            
                    new_rate = pd.DataFrame(np.full((rate_sorted.shape[0], len(all_stm_unique)*n_samples), np.nan), columns=np.repeat(all_stm_unique, n_samples)).astype('float32')
                    for trial_type_ind, trial_type in enumerate(all_stm_unique):
                        # match noise correlation using Gaussian copula
                        if rescaled_r.shape[1] == 1: # one r for each neuron
                            marginal_params = np.stack([rescaled_r[:, 0], new_p[:, trial_type_ind]]).T # num_neurons x 2 (r & p)
                        else: # one r for each combination of neuron and stimulus
                            marginal_params = np.stack([rescaled_r[:, trial_type_ind], new_p[:, trial_type_ind]]).T # num_neurons x 2 (r & p)
                        new_rate.loc[:, trial_type] = sample_nb_with_copula(rate_sorted.loc[:, trial_type], marginal_params, n_samples=n_samples, seed=sess_ind*100, model=model)

                    list_rate_nb[f_ind] = new_rate.copy()
                    # print(f'sess_ind {sess_ind}, f_ind {f_ind} duration {(time() - start_time)/60:.2f} min')
        
        elif model == 'poisson':
            method = 'rcomb'
            use_rrep = True
            lam_estim_neu = rate_sorted_mean_coll.values.copy() # mean-matched poisson
            lam_estim_neu = rate_sorted_var_coll.values.copy() # var-matched poisson

            start_time = time()
            new_rate = pd.DataFrame(np.full((rate_sorted.shape[0], len(all_stm_unique)*n_samples), np.nan), columns=np.repeat(all_stm_unique, n_samples)).astype('float32')
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                if trial_type_ind % 60 == 0:
                    print(f'sess_ind {sess_ind}, trial_type_ind {trial_type_ind}')
                    
                # match noise correlation using Gaussian copula
                marginal_params = lam_estim_neu[:, trial_type_ind].copy() # num_neurons
                new_rate.loc[:, trial_type] = sample_nb_with_copula(rate_sorted.loc[:, trial_type], marginal_params, n_samples=n_samples, seed=sess_ind*100, model=model)

            list_rate_poiss = new_rate.copy()
            print(f'sess_ind {sess_ind} duration {(time() - start_time)/60:.2f} min')
                    
    # Save into a file
    if not use_rrep: # one r for each combination of neuron and stimulus
        method += '_rcomb'
    filename = 'rate_nb_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_' + str(sess_ind) + '.pickle'
    # filename = 'rate_poiss_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': 'list_rate_nb', 'list_rate_nb': list_rate_nb}, f)
        # pickle.dump({'tree_variables': 'list_rate_poiss', 'list_rate_poiss': list_rate_poiss}, f)
           
    print("Ended Process",c_proc.name)

# %%
# sample spike count from discrete probability distributions (do not use Gaussian copula)
def sample_spkcnt_indep(sess_ind, n_samples, model='katz'):

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        
        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

        print(f'session index: {sess_ind}')
        
        rate_sorted = list_rate_all[sess_ind].sort_index(axis=1)
        stm = rate_sorted.columns.copy()
        num_neurons = rate_sorted.shape[0]

        # Multiply by delta t to convert to spike counts
        rate_sorted = rate_sorted * 0.25
        # print(rate_sorted.shape[0])

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

        # Compute mean & variance for each stimulus
        rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
        rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

        list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()

        if model == 'katz': # nb + poisson + binomial
            start_time = time()
            new_rate = pd.DataFrame(np.full((rate_sorted.shape[0], len(all_stm_unique)*n_samples), np.nan), columns=np.repeat(all_stm_unique, n_samples)).astype('float32')
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                if trial_type_ind % 60 == 0:
                    print(f'sess_ind {sess_ind}, trial_type_ind {trial_type_ind}')
                r_estim_neu = np.full(rate_sorted.shape[0], np.nan)
                p_estim_neu = np.full(rate_sorted.shape[0], np.nan)
                lam_estim_neu = np.full(rate_sorted.shape[0], np.nan)
                n_estim_neu = np.full(rate_sorted.shape[0], np.nan)
                
                mean_tt, var_tt = rate_sorted_mean_coll.loc[:, trial_type].values.copy(), rate_sorted_var_coll.loc[:, trial_type].values.copy()
                margin = 0.05
                bool_nb = var_tt > mean_tt*(1+margin)
                bool_poiss = (var_tt >= mean_tt*(1-margin)) & (var_tt < mean_tt*(1+margin))
                r_estim_neu[bool_nb] = mean_tt[bool_nb]**2 / (var_tt[bool_nb] - mean_tt[bool_nb])
                p_estim_neu[bool_nb] = mean_tt[bool_nb] / var_tt[bool_nb]
                lam_estim_neu[bool_poiss] = mean_tt[bool_poiss].copy()
                bool_bin = var_tt < mean_tt*(1-margin)
                n_estim_neu[bool_bin] = mean_tt[bool_bin]**2 / (mean_tt[bool_bin] - var_tt[bool_bin])
                p_estim_neu[bool_bin] = 1 - var_tt[bool_bin] / mean_tt[bool_bin]

                # do not match noise correlation
                marginal_params = np.stack([r_estim_neu, p_estim_neu, lam_estim_neu, n_estim_neu]).T # num_neurons x 4 (r, p, lam, n)
                new_rate.loc[:, trial_type] = sample_nb_indep(rate_sorted.loc[:, trial_type], marginal_params, sess_ind, trial_type_ind,
                                                              n_samples=n_samples, model=model, margin=margin)

            list_rate_katz_asis = new_rate.copy()
            print(f'sess_ind {sess_ind}, as-is duration {(time() - start_time)/60:.2f} min')

            list_rate_katz_RRneuron = np.full((len(list_target_slopes), rate_sorted.shape[0], len(all_stm_unique)*n_samples), np.nan)
            for slope_ind, target_slope in enumerate(list_target_slopes):
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
            
                start_time = time()
                new_rate_RRneuron = pd.DataFrame(np.full((rate_RRneuron_dr.shape[0], len(all_stm_unique)*n_samples), np.nan), columns=np.repeat(all_stm_unique, n_samples)).astype('float32')
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    if trial_type_ind % 60 == 0:
                        print(f'sess_ind {sess_ind}, trial_type_ind {trial_type_ind}')
                    r_estim_neu = np.full(rate_RRneuron_dr.shape[0], np.nan)
                    p_estim_neu = np.full(rate_RRneuron_dr.shape[0], np.nan)
                    lam_estim_neu = np.full(rate_RRneuron_dr.shape[0], np.nan)
                    n_estim_neu = np.full(rate_RRneuron_dr.shape[0], np.nan)
                    
                    mean_tt, var_tt = rate_mean_RRneuron_coll.loc[:, trial_type].values.copy(), rate_var_RRneuron_coll.loc[:, trial_type].values.copy()
                    margin = 0.05
                    bool_nb = var_tt > mean_tt*(1+margin)
                    bool_poiss = (var_tt >= mean_tt*(1-margin)) & (var_tt < mean_tt*(1+margin))
                    # bool_poiss = ~bool_nb
                    r_estim_neu[bool_nb] = mean_tt[bool_nb]**2 / (var_tt[bool_nb] - mean_tt[bool_nb])
                    p_estim_neu[bool_nb] = mean_tt[bool_nb] / var_tt[bool_nb]
                    lam_estim_neu[bool_poiss] = mean_tt[bool_poiss].copy()
                    bool_bin = var_tt < mean_tt*(1-margin)
                    n_estim_neu[bool_bin] = mean_tt[bool_bin]**2 / (mean_tt[bool_bin] - var_tt[bool_bin])
                    p_estim_neu[bool_bin] = 1 - var_tt[bool_bin] / mean_tt[bool_bin]

                    # do not match noise correlation
                    marginal_params = np.stack([r_estim_neu, p_estim_neu, lam_estim_neu, n_estim_neu]).T # num_neurons x 4 (r, p, lam, n)
                    new_rate_RRneuron.loc[:, trial_type] = sample_nb_indep(rate_RRneuron_dr.loc[:, trial_type], marginal_params, sess_ind, trial_type_ind,
                                                                           n_samples=n_samples, model=model, margin=margin)                  
                
                list_rate_katz_RRneuron[slope_ind] = new_rate_RRneuron.copy()
                print(f'sess_ind {sess_ind}, target_slope {target_slope:.1f} duration {(time() - start_time)/60:.2f} min')
                    
    # Save into a file
    filename = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_rate_katz_asis', 'list_rate_katz_RRneuron'],
                     'list_rate_katz_asis': list_rate_katz_asis, 'list_rate_katz_RRneuron': list_rate_katz_RRneuron}, f)
           
    print("Ended Process",c_proc.name)

# %%
# decoding (ABO)
def decode_ABO(sess_ind, decoder_type, method='resc_r', n_samples=50, use_rrep=True):

    ''' decoder_type is SVM, logit, RF, kNN '''

    # ignore warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        list_target_slopes = [10, 1, 0]
        # list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
        num_trial_types = 119
        n_splits = 10

        np.random.seed(0)

        all_stimuli = np.arange(-1, 118, 1).astype(int) # include grayscreen
        train_stimuli = all_stimuli.copy()

        rate = list_rate_all[sess_ind].copy()
        
        # print(f'sess_ind: {sess_ind}')

        # rate_nb loading
        
        # NB+P
        if not use_rrep: # one r for each combination of neuron and stimulus
            method += '_rcomb'
        file_name = 'rate_nb_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_realR_all.pickle' # r=inf when var_G=0 (500 samples), r=inf when var<mean (50 samples), r=inf when var<mean (500/50 samples, rcomb)
        with open(file_name, 'rb') as f:
            rate_nb_copula_ABO = pickle.load(f)
            list_rate_nb2_rep = rate_nb_copula_ABO['list_rate_nb2'][sess_ind].copy()

        # # poisson
        # list_rate_nb2_rep = np.full((len(list_target_slopes), rate.shape[0], num_trial_types*n_samples), np.nan)
        # method_poiss = 'rcomb'
        # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all3.pickle' # lam=mean for comb of neu/stim (500/50 samples)
        # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all4.pickle' # lam=var for comb of neu/stim (500/50 samples)
        # with open(file_name, 'rb') as f:
        #     rate_poiss_copula_ABO = pickle.load(f)
        #     list_rate_nb2_rep[0] = rate_poiss_copula_ABO['list_rate_poiss2'][sess_ind].copy()
        # method = method_poiss

        # # katz (nb + poisson + binomial)
        # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all.pickle' # slope 0-2, poiss margin 0.05, binom n>=1 (50 samples)
        # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all2.pickle' # slope 0-2, poiss margin 0.05, binom n>=1, use random_state (50 samples)
        # with open(file_name, 'rb') as f:
        #     rate_nb_copula_ABO = pickle.load(f)
        #     rate = rate_nb_copula_ABO['list_rate_katz_asis'][sess_ind].copy()
        #     list_rate_RRneuron_dr = rate_nb_copula_ABO['list_rate_katz_RRneuron2'][sess_ind].copy()

        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # Multiply by delta t to convert to spike counts (comment this part when using sampled spike count as 'rate')
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

        # trial order re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        # Re-convert to 2D response matrix
        label_train = np.repeat(all_stimuli, min_num_trials)
        # rate_train = pd.DataFrame(rate_sorted.reshape(rate_sorted.shape[0], -1), columns=label_train)
        rate_train = rate_sorted.reshape(rate_sorted.shape[0], -1)
        
        # decoding cross-validation (as-is)
        kfold = KFold(n_splits=n_splits)
        stkfold = StratifiedKFold(n_splits=n_splits)

        list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
        list_accuracy = np.full(n_splits, np.nan)

        for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
        # for probe_ind, probe_stim in enumerate(all_stimuli):
            start_time = time()
            # X_train, X_test = rate_train.T.iloc[train_index].copy(), rate_train.T.iloc[test_index].copy() # train, test data/label
            # y_train, y_test = label_train[train_index].copy(), label_train[test_index].copy()

            # mean_ = X_train.mean(axis=0)
            # X_train = X_train.sub(mean_, axis=1) # train data mean centering
            # X_test = X_test.sub(mean_, axis=1)

            X_train, X_test = rate_train.T[train_index].copy(), rate_train.T[test_index].copy() # train, test data/label
            y_train, y_test = label_train[train_index].copy(), label_train[test_index].copy()
            mean_ = X_train.mean(axis=0)
            X_train = X_train - mean_ # train data mean centering
            X_test = X_test - mean_

            if decoder_type == 'SVM':
                clf = SVC(kernel='linear', max_iter=1000)
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

            print(f'session {sess_ind}, as-is, split_ind {split_ind}, duration {(time()-start_time)/60:.2f} min')

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
            if slope_ind == 1: # 1 for NB+P, 0 for poisson; comment this part for katz
                # start_time = time()
                print(f'session {sess_ind}, target_slope {target_slope}')
                
                # Re-convert to 2D response matrix
                label_train_RRneuron = np.repeat(all_stimuli, n_samples)
                rate_train_RRneuron = pd.DataFrame(list_rate_nb2_rep[slope_ind], columns=label_train_RRneuron).copy()
                # rate_train_RRneuron = pd.DataFrame(list_rate_RRneuron_dr[slope_ind], columns=label_train_RRneuron).copy()
                # rate_train_RRneuron = list_rate_RRneuron_dr[slope_ind].copy()

                # decoding cross-validation (as-is)
                kfold = KFold(n_splits=n_splits)
                stkfold = StratifiedKFold(n_splits=n_splits)

                list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
                list_accuracy = np.full(n_splits, np.nan)
                for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
                # for probe_ind, probe_stim in enumerate(all_stimuli):
                    start_time = time()
                    X_train, X_test = rate_train_RRneuron.T.iloc[train_index].copy(), rate_train_RRneuron.T.iloc[test_index].copy() # train, test data/label
                    y_train, y_test = label_train_RRneuron[train_index].copy(), label_train_RRneuron[test_index].copy()

                    mean_ = X_train.mean(axis=0)
                    X_train = X_train.sub(mean_, axis=1) # train data mean centering
                    X_test = X_test.sub(mean_, axis=1)

                    # X_train, X_test = rate_train_RRneuron.T[train_index].copy(), rate_train_RRneuron.T[test_index].copy() # train, test data/label
                    # y_train, y_test = label_train_RRneuron[train_index].copy(), label_train_RRneuron[test_index].copy()
                    # mean_ = X_train.mean(axis=0)
                    # X_train = X_train - mean_ # train data mean centering
                    # X_test = X_test - mean_
        
                    if decoder_type == 'SVM':
                        clf = SVC(kernel='linear', max_iter=1000)
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

                    print(f'session {sess_ind}, target_slope {target_slope:.1f}, split_ind {split_ind}, duration {(time()-start_time)/60:.2f} min')

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
    filename = decoder_type + '_decoding_ABO_allstim_nb_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    # filename = decoder_type + '_decoding_ABO_allstim_poiss_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    # filename = decoder_type + '_decoding_ABO_allstim_katz_indep_sep_RRneuron_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['mean_confusion_test_asis', 'mean_accuracy_asis', 'list_mean_confusion_test_RRneuron', 'list_mean_accuracy_RRneuron'],
                     'mean_confusion_test_asis': mean_confusion_test_asis, 'mean_accuracy_asis': mean_accuracy_asis,
                     'list_mean_confusion_test_RRneuron': list_mean_confusion_test_RRneuron, 'list_mean_accuracy_RRneuron': list_mean_accuracy_RRneuron}, f)
        # pickle.dump({'tree_variables': ['mean_confusion_test_asis', 'mean_accuracy_asis'],
        #              'mean_confusion_test_asis': mean_confusion_test_asis, 'mean_accuracy_asis': mean_accuracy_asis}, f)
                
    print("Ended Process", c_proc.name)

# %%
# overlap of stimulus pairs
def compute_overlap_stimpairs(sess_ind, method='resc_r', n_samples=50, use_rrep=True):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = [10, 1, 0]
    # list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    rate = list_rate_all[sess_ind].copy()

    # rate_nb loading
    
    # NB+P
    if not use_rrep: # one r for each combination of neuron and stimulus
        method += '_rcomb'
    file_name = 'rate_nb_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_realR_all.pickle' # r=inf when var_G=0 (500 samples), r=inf when var<mean (50 samples), r=inf when var<mean (500/50 samples, rcomb)
    with open(file_name, 'rb') as f:
        rate_nb_copula_ABO = pickle.load(f)
        list_rate_nb2_rep = rate_nb_copula_ABO['list_rate_nb2'][sess_ind].copy()

    # # poisson
    # list_rate_nb2_rep = np.full((len(list_target_slopes), rate.shape[0], num_trial_types*n_samples), np.nan)
    # method_poiss = 'rcomb'
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all3.pickle' # lam=mean for comb of neu/stim (500/50 samples)
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all4.pickle' # lam=var for comb of neu/stim (500/50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_poiss_copula_ABO = pickle.load(f)
    #     list_rate_nb2_rep[0] = rate_poiss_copula_ABO['list_rate_poiss2'][sess_ind].copy()
    # method = method_poiss

    # # katz (nb + poisson + binomial)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all.pickle' # slope 0-2, poiss margin 0.05, binom n>=1 (50 samples)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all2.pickle' # slope 0-2, poiss margin 0.05, binom n>=1, use random_state (50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_nb_copula_ABO = pickle.load(f)
    #     rate = rate_nb_copula_ABO['list_rate_katz_asis'][sess_ind].copy()
    #     list_rate_RRneuron_dr = rate_nb_copula_ABO['list_rate_katz_RRneuron2'][sess_ind].copy()

    print(f'sess_ind: {sess_ind}')
    
    rate_sorted = rate.sort_index(axis=1)
    stm = rate_sorted.columns.copy()
    num_neurons = rate_sorted.shape[0]

    # Multiply by delta t to convert to spike counts (comment this part when using sampled spike count as 'rate')
    rate_sorted = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()

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
    
    list_overlap_RRneuron2 = np.full((len(list_target_slopes), num_trial_types, num_trial_types), np.nan)
    list_gap_RRneuron2 = np.full((len(list_target_slopes), num_trial_types, num_trial_types), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        if slope_ind == 1: # 1 for NB+P, 0 for poisson; comment this part for katz
            start_time = time()
            # print(f'target slope = {target_slope:.1f}')

            rate_RRneuron_dr = pd.DataFrame(list_rate_nb2_rep[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
            # rate_RRneuron_dr = pd.DataFrame(list_rate_RRneuron_dr[slope_ind], columns=np.repeat(all_stm_unique, n_samples))

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
            print(f'sess_ind {sess_ind}, target_slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # strongly connected component (SCC)

    # convert overlap matrix digonal to 0
    list_overlap_asis[np.eye(num_trial_types, dtype=bool)] = 0
    for slope_ind, target_slope in enumerate(list_target_slopes):
        list_overlap_RRneuron2[slope_ind, np.eye(num_trial_types, dtype=bool)] = 0

    G_dir = nx.from_numpy_array(list_overlap_asis, create_using=nx.DiGraph)
    sccs = nx.strongly_connected_components(G_dir)
    largest_scc = max(sccs, key=len)
    G_scc = G_dir.subgraph(largest_scc).copy()
    size_scc_asis = len(list(G_scc.nodes))

    list_size_scc_RRneuron = np.full(len(list_target_slopes), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        if slope_ind == 1: # 1 for NB+P, 0 for poisson; comment this part for katz
            G_dir = nx.from_numpy_array(list_overlap_RRneuron2[slope_ind], create_using=nx.DiGraph)
            sccs = nx.strongly_connected_components(G_dir)
            largest_scc = max(sccs, key=len)
            G_scc = G_dir.subgraph(largest_scc).copy()
            list_size_scc_RRneuron[slope_ind] = len(list(G_scc.nodes))

    # Save into a file
    filename = 'overlap_nbr_stimpairs_ABO_nb_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    # filename = 'overlap_nbr_stimpairs_ABO_poiss_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    # filename = 'overlap_nbr_stimpairs_ABO_katz_indep_sep_RRneuron_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    with open(filename, "wb") as f:
        # pickle.dump({'tree_variables': ['list_overlap_asis', 'list_overlap_RRneuron2', 'list_gap_asis', 'list_gap_RRneuron2'],
        #              'list_overlap_asis': list_overlap_asis, 'list_overlap_RRneuron2': list_overlap_RRneuron2,
        #              'list_gap_asis': list_gap_asis, 'list_gap_RRneuron2': list_gap_RRneuron2}, f)
        pickle.dump({'tree_variables': ['list_overlap_asis', 'list_overlap_RRneuron2', 'size_scc_asis', 'list_size_scc_RRneuron'],
                     'list_overlap_asis': list_overlap_asis, 'list_overlap_RRneuron2': list_overlap_RRneuron2,
                     'size_scc_asis': size_scc_asis, 'list_size_scc_RRneuron': list_size_scc_RRneuron}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA across session pairs (ABO Neuropixels, RRneuron)
def RSA_across_sesspairs_ABO(sess_ind, similarity_type, method='resc_r', n_samples=50, use_rrep=True):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.array([10, 1, 0])
    # list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    shift_factor = 0
    random_state = 0
    num_trial_types = 119

    print(f'sess_ind {sess_ind}')

    np.random.seed(0) # match trial order.

    rate = list_rate_all[sess_ind].copy()

    # rate_nb loading
    
    # NB+P
    if not use_rrep: # one r for each combination of neuron and stimulus
        method += '_rcomb'
    file_name = 'rate_nb_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_realR_all.pickle' # r=inf when var_G=0 (500 samples), r=inf when var<mean (50 samples), r=inf when var<mean (500/50 samples, rcomb)
    with open(file_name, 'rb') as f:
        rate_nb_copula_ABO = pickle.load(f)
        list_rate_nb2_rep = rate_nb_copula_ABO['list_rate_nb2'][sess_ind].copy()

    # # poisson
    # list_rate_nb2_rep = np.full((len(list_target_slopes), rate.shape[0], num_trial_types*n_samples), np.nan)
    # method_poiss = 'rcomb'
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all3.pickle' # lam=mean for comb of neu/stim (500/50 samples)
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all4.pickle' # lam=var for comb of neu/stim (500/50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_poiss_copula_ABO = pickle.load(f)
    #     list_rate_nb2_rep[0] = rate_poiss_copula_ABO['list_rate_poiss2'][sess_ind].copy()
    # method = method_poiss

    # # katz (nb + poisson + binomial)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all.pickle' # slope 0-2, poiss margin 0.05, binom n>=1 (50 samples)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all2.pickle' # slope 0-2, poiss margin 0.05, binom n>=1, use random_state (50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_nb_copula_ABO = pickle.load(f)
    #     rate = rate_nb_copula_ABO['list_rate_katz_asis'][sess_ind].copy()
    #     list_rate_katz_RRneuron = rate_nb_copula_ABO['list_rate_katz_RRneuron2'][sess_ind].copy()

    # rate_sorted = rate.sort_index(axis=1)
    stm = rate.columns.copy()
    num_neurons = rate.shape[0]

    # Multiply by delta t to convert to spike counts (comment this part when using sampled spike count as 'rate')
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
    
    list_RSM_mean_RRneuron = np.zeros((len(list_target_slopes), num_trial_types, num_trial_types))
    list_rate_RRneuron_dr = np.empty(len(list_target_slopes), dtype=object)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        if slope_ind == 1: # 1 for NB+P, 0 for poisson; comment this part for katz
            start_time = time()

            rate_RRneuron_dr = pd.DataFrame(list_rate_nb2_rep[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
            # rate_RRneuron_dr = pd.DataFrame(list_rate_katz_RRneuron[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
            list_rate_tt = [None] * num_trial_types
            for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
                list_rate_tt[trial_type_ind] = rate_RRneuron_dr.loc[:, trial_type].iloc[:, :min_num_trials].copy()
            rate_RRneuron_dr = np.stack(list_rate_tt, axis=2)
            rate_RRneuron_dr = np.transpose(rate_RRneuron_dr, (0, 2, 1)) # num_neurons x num_trial_types x num_trials

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
            print(f'sess_ind {sess_ind}, target_slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'RSM_ABO_allneu_nb_sep_' + similarity_type  + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    # filename = 'RSM_ABO_allneu_poiss_sep_' + similarity_type  + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    # filename = 'RSM_ABO_allneu_katz_indep_sep_RRneuron_' + similarity_type + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis', 'list_rate_RRneuron_dr', 'list_RSM_mean_RRneuron'], \
                     'list_RSM_mean_asis': list_RSM_mean_asis, 'list_rate_RRneuron_dr': list_rate_RRneuron_dr, 'list_RSM_mean_RRneuron': list_RSM_mean_RRneuron}, f)
        # pickle.dump({'tree_variables': ['list_RSM_mean_asis', 'list_RSM_mean_RRneuron'], \
        #              'list_RSM_mean_asis': list_RSM_mean_asis, 'list_RSM_mean_RRneuron': list_RSM_mean_RRneuron}, f)
        # pickle.dump({'tree_variables': 'list_RSM_mean_asis', \
        #              'list_RSM_mean_asis': list_RSM_mean_asis}, f)
        
    print("Ended Process", c_proc.name)

# %%
# RSA within sessions (ABO, RRneuron)
def RSA_withinsess_ABO(sess_ind, similarity_type, method='resc_r', n_samples=50, use_rrep=True):
    
    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.array([10, 1, 0])
    # list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trials = 50
    num_trial_types = 119
    n_neu_sampling = 10

    print(f'sess_ind {sess_ind}')

    np.random.seed(0) # match neuron partitioning
    # random.seed(0)

    rate = list_rate_all[sess_ind].copy()

    # rate_nb loading
    
    # NB+P
    if not use_rrep: # one r for each combination of neuron and stimulus
        method += '_rcomb'
    file_name = 'rate_nb_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_realR_all.pickle' # r=inf when var_G=0 (500 samples), r=inf when var<mean (50 samples), r=inf when var<mean (500/50 samples, rcomb)
    with open(file_name, 'rb') as f:
        rate_nb_copula_ABO = pickle.load(f)
        list_rate_nb2_rep = rate_nb_copula_ABO['list_rate_nb2'][sess_ind].copy()

    # # poisson
    # list_rate_nb2_rep = np.full((len(list_target_slopes), rate.shape[0], num_trial_types*n_samples), np.nan)
    # method_poiss = 'rcomb'
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all3.pickle' # lam=mean for comb of neu/stim (500/50 samples)
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all4.pickle' # lam=var for comb of neu/stim (500/50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_poiss_copula_ABO = pickle.load(f)
    #     list_rate_nb2_rep[0] = rate_poiss_copula_ABO['list_rate_poiss2'][sess_ind].copy()
    # method = method_poiss

    # # katz (nb + poisson + binomial)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all.pickle' # slope 0-2, poiss margin 0.05, binom n>=1 (50 samples)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all2.pickle' # slope 0-2, poiss margin 0.05, binom n>=1, use random_state (50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_nb_copula_ABO = pickle.load(f)
    #     rate = rate_nb_copula_ABO['list_rate_katz_asis'][sess_ind].copy()
    #     list_rate_katz_RRneuron = rate_nb_copula_ABO['list_rate_katz_RRneuron2'][sess_ind].copy()

    list_corr_withinsess_asis = np.full((n_neu_sampling, 3), np.nan)
    list_corr_withinsess2 = np.full((len(list_target_slopes), n_neu_sampling, 3), np.nan)

    # rate_sorted = rate.sort_index(axis=1)
    stm = rate.columns.copy()
    num_neurons = rate.shape[0]

    # Multiply by delta t to convert to spike counts (comment this part when using sampled spike count as 'rate')
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

    list_rate_RRneuron_dr = np.full((len(list_target_slopes), *rate_sorted.shape), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        rate_RRneuron_dr = pd.DataFrame(list_rate_nb2_rep[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
        # rate_RRneuron_dr = pd.DataFrame(list_rate_katz_RRneuron[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
            list_rate_tt[trial_type_ind] = rate_RRneuron_dr.loc[:, trial_type].iloc[:, :min_num_trials].copy()
        rate_RRneuron_dr = np.stack(list_rate_tt, axis=2)
        rate_RRneuron_dr = np.transpose(rate_RRneuron_dr, (0, 2, 1)) # num_neurons x num_trial_types x num_trials

        list_rate_RRneuron_dr[slope_ind] = rate_RRneuron_dr.copy()

    # Iterate over neuron partitionings
    list_RSM_neu1_all = np.full((n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu2_all = np.full((n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu1_RRneuron_all = np.full((len(list_target_slopes), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    list_RSM_neu2_RRneuron_all = np.full((len(list_target_slopes), n_neu_sampling, num_trial_types, num_trial_types), np.nan)
    for neu_sample_ind in range(n_neu_sampling):
        print(f'neu_sample_ind = {neu_sample_ind}')
        
        # Partition neurons
        neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
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

        list_corr_withinsess_asis[neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
        bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
        list_corr_withinsess_asis[neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
        list_corr_withinsess_asis[neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
    
        for slope_ind, target_slope in enumerate(list_target_slopes):
            if slope_ind == 1: # 1 for NB+P, 0 for poisson; comment this part for katz
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

                start_time = time()
                count = 0
                for sampling_ind in range(n_sampling):
                    rate_sampled_trials1_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy()
                    rate_sampled_trials1_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                    rate_sampled_trials2_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy()
                    rate_sampled_trials2_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                    RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2))
                    RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2))

                    list_RSM_neu1[sampling_ind] = RSM1.copy()
                    list_RSM_neu2[sampling_ind] = RSM2.copy()

                    count += 1
                    if count % (n_sampling//2) == 0:
                        print(f'count: {count}')
                print(f'sess_ind {sess_ind}, neu_sample_ind {neu_sample_ind}, target_slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

                RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
                RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                list_RSM_neu1_RRneuron_all[slope_ind, neu_sample_ind], list_RSM_neu2_RRneuron_all[slope_ind, neu_sample_ind] = RSM_mean_neu1.copy(), RSM_mean_neu2.copy()

                list_corr_withinsess2[slope_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_withinsess2[slope_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_withinsess2[slope_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # Save into a file
    filename = 'RSM_corr_withinsess_ABO_nb_sep_' + similarity_type + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    # filename = 'RSM_corr_withinsess_ABO_poiss_sep_' + similarity_type + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    # filename = 'RSM_corr_withinsess_ABO_katz_indep_sep_RRneuron_' + similarity_type + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) +  '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_neu1_all', 'list_RSM_neu2_all', 'list_RSM_neu1_RRneuron_all', 'list_RSM_neu2_RRneuron_all', 'list_corr_withinsess_asis', 'list_corr_withinsess2'], \
                    'list_RSM_neu1_all': list_RSM_neu1_all, 'list_RSM_neu2_all': list_RSM_neu2_all,
                    'list_RSM_neu1_RRneuron_all': list_RSM_neu1_RRneuron_all, 'list_RSM_neu2_RRneuron_all': list_RSM_neu2_RRneuron_all,
                    'list_corr_withinsess_asis': list_corr_withinsess_asis, 'list_corr_withinsess2': list_corr_withinsess2}, f)
        # pickle.dump({'tree_variables': ['list_RSM_neu1_all', 'list_RSM_neu2_all', 'list_corr_withinsess_asis'], \
        #             'list_RSM_neu1_all': list_RSM_neu1_all, 'list_RSM_neu2_all': list_RSM_neu2_all,
        #             'list_corr_withinsess_asis': list_corr_withinsess_asis}, f)
        
    print("Ended Process", c_proc.name)

# %%
# Effective dimensionality
def compute_eff_dim(sess_ind, method='resc_r', n_samples=50, use_rrep=True, n_trial_sampling=10):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = [10, 1, 0]
    # list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119

    print(f'sess_ind {sess_ind}')
    rng = np.random.default_rng(sess_ind)

    list_dim_asis = np.zeros(num_trial_types)
    list_dim_nb = np.zeros((len(list_target_slopes), num_trial_types))
    list_dim_global_asis = np.zeros(2)
    list_dim_global_nb = np.zeros((len(list_target_slopes), 2))
    list_dim_sam_asis = np.zeros(n_trial_sampling)
    list_dim_sam_nb = np.zeros((len(list_target_slopes), n_trial_sampling))

    rate = list_rate_all[sess_ind].copy()
    
    # rate_nb loading
    
    # NB+P
    if not use_rrep: # one r for each combination of neuron and stimulus
        method += '_rcomb'
    file_name = 'rate_nb_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_realR_all.pickle' # r=inf when var_G=0 (500 samples), r=inf when var<mean (50 samples), r=inf when var<mean (500/50 samples, rcomb)
    with open(file_name, 'rb') as f:
        rate_nb_copula_ABO = pickle.load(f)
        list_rate_nb2_rep = rate_nb_copula_ABO['list_rate_nb2'][sess_ind].copy()

    # # poisson
    # list_rate_nb2_rep = np.full((len(list_target_slopes), rate.shape[0], num_trial_types*n_samples), np.nan)
    # method_poiss = 'rcomb'
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all3.pickle' # lam=mean for comb of neu/stim (500/50 samples)
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all4.pickle' # lam=var for comb of neu/stim (500/50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_poiss_copula_ABO = pickle.load(f)
    #     list_rate_nb2_rep[0] = rate_poiss_copula_ABO['list_rate_poiss2'][sess_ind].copy()
    # method = method_poiss

    # # katz (nb + poisson + binomial)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all.pickle' # slope 0-2, poiss margin 0.05, binom n>=1 (50 samples)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all2.pickle' # slope 0-2, poiss margin 0.05, binom n>=1, use random_state (50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_nb_copula_ABO = pickle.load(f)
    #     rate = rate_nb_copula_ABO['list_rate_katz_asis'][sess_ind].copy()
    #     list_rate_RRneuron_dr = rate_nb_copula_ABO['list_rate_katz_RRneuron2'][sess_ind].copy()
    
    rate_sorted = rate.sort_index(axis=1)
    stm = rate_sorted.columns.copy()

    # Multiply by delta t to convert to spike counts (comment this part when using sampled spike count as 'rate')
    rate = rate * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

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
        
    for slope_ind, target_slope in enumerate(list_target_slopes):
        if slope_ind == 1: # 1 for NB+P, 0 for poisson; comment this part for katz
            rate_nb = pd.DataFrame(list_rate_nb2_rep[slope_ind], columns=np.repeat(all_stm_unique, n_samples)).copy()
            # rate_nb = pd.DataFrame(list_rate_RRneuron_dr[slope_ind], columns=np.repeat(all_stm_unique, n_samples)).copy()
        
            # Compute mean & variance for each stimulus
            stm_cnt_dict = dict(zip(all_stm_unique, np.repeat([n_samples], len(all_stm_unique))))
            rate_sorted_mean_nb, rate_sorted_var_nb = compute_mean_var_trial(stm_cnt_dict, rate_nb)
            rate_sorted_mean_coll_nb, rate_sorted_var_coll_nb = compute_mean_var_trial_collapse(stm_cnt_dict, rate_nb)

            # pca
            n_components = rate_nb.shape[0]
            pca = PCA(n_components=n_components)

            # Compute effective dimensionality for each stimulus
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                if trial_type_ind % 60 == 0:
                    print(f'sess_ind {sess_ind}, target_slope {target_slope:.1f}, trial_type_ind {trial_type_ind}')
                cf = np.cov(rate_nb.loc[:, trial_type])
                list_dim_nb[slope_ind, trial_type_ind] = ((np.trace(cf)**2) / np.trace(cf @ cf)) / rate_nb.shape[0]
            cf_all = np.cov(rate_nb)
            list_dim_global_nb[slope_ind, 0] = ((np.trace(cf_all)**2) / np.trace(cf_all @ cf_all)) / rate_nb.shape[0]
            cf_cen = np.cov(rate_sorted_mean_coll_nb)
            list_dim_global_nb[slope_ind, 1] = ((np.trace(cf_cen)**2) / np.trace(cf_cen @ cf_cen)) / rate_nb.shape[0]

            rand_tt_inds = rng.permutation(range(rate_nb.shape[1]))
            rate_nb = rate_nb.iloc[:, rand_tt_inds]
            for t_sam_ind in range(n_trial_sampling):
                rate_sam = np.full_like(rate_sorted_mean_coll_nb, np.nan)
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    rate_tt = rate_nb.loc[:, trial_type].copy()
                    rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
                cf_sam = np.cov(rate_sam)
                list_dim_sam_nb[slope_ind, t_sam_ind] = ((np.trace(cf_sam)**2) / np.trace(cf_sam @ cf_sam)) / rate_nb.shape[0]

    # Save into a file
    filename = 'eff_dim_DC_ABO_nb_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    # filename = 'eff_dim_DC_ABO_poiss_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    # filename = 'eff_dim_DC_ABO_katz_indep_sep_RRneuron_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        # pickle.dump({'tree_variables': 'list_dim_nb', 'list_dim_nb': list_dim_nb}, f)
        pickle.dump({'tree_variables': ['list_dim_nb', 'list_dim_global_nb', 'list_dim_sam_nb'],
                     'list_dim_nb': list_dim_nb, 'list_dim_global_nb': list_dim_global_nb, 'list_dim_sam_nb': list_dim_sam_nb}, f)
        # pickle.dump({'tree_variables': ['list_dim_asis', 'list_dim_nb', 'list_dim_global_asis', 'list_dim_global_nb', 'list_dim_sam_asis', 'list_dim_sam_nb'],
        #             'list_dim_asis': list_dim_asis, 'list_dim_nb': list_dim_nb, 'list_dim_global_asis': list_dim_global_asis, 'list_dim_global_nb': list_dim_global_nb,
        #             'list_dim_sam_asis': list_dim_sam_asis, 'list_dim_sam_nb': list_dim_sam_nb}, f)
        # pickle.dump({'tree_variables': ['list_dim_asis', 'list_dim_global_asis', 'list_dim_sam_asis'],
        #             'list_dim_asis': list_dim_asis, 'list_dim_global_asis': list_dim_global_asis,
        #             'list_dim_sam_asis': list_dim_sam_asis}, f)

    print("Ended Process", c_proc.name)

# %%
# slope and intercept
def linreg_nb(sess_ind, n_samples=50):

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    num_trial_types = 119
    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    
    # print(f'sess_ind {sess_ind}')

    # rate = list_rate_all[sess_ind]

    # katz (nb + poisson + binomial)
    file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all.pickle' # slope 0-2, poiss margin 0.05, binom n>=1 (50 samples)
    file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all2.pickle' # slope 0-2, poiss margin 0.05, binom n>=1, use random_state (50 samples)
    with open(file_name, 'rb') as f:
        rate_nb_copula_ABO = pickle.load(f)
        rate = rate_nb_copula_ABO['list_rate_katz_asis'][sess_ind].copy()
        list_rate_RRneuron_dr = rate_nb_copula_ABO['list_rate_katz_RRneuron2'][sess_ind].copy()

    rate_sorted = rate.sort_index(axis=1)
    stm = rate_sorted.columns.copy()

    # # Multiply by delta t to convert to spike counts (comment this part when using sampled spike count as 'rate')
    # rate_sorted = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns).copy()

    # Calculate and collect linear slopes for all stimuli
    list_slopes_all_an_loglog_nbpb_asis = np.zeros((2, rate_sorted_mean_coll.shape[1]))
    for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):
        # loglog scale
        bool_mean_nb_notzero = rate_sorted_mean_coll.loc[:, trial_type] > 0
        popt = np.polyfit(np.log10(rate_sorted_mean_coll.loc[bool_mean_nb_notzero, trial_type].values).flatten().astype(np.float32), \
                            np.log10(rate_sorted_var_coll.loc[bool_mean_nb_notzero, trial_type].values).flatten().astype(np.float32), 1) # 모든 trial type의 trial 수 같으므로 collapsed 버전 사용
        list_slopes_all_an_loglog_nbpb_asis[:, trial_type_ind] = popt.copy()

    list_slopes_all_an_loglog_nbpb = np.full((len(list_target_slopes), 2, num_trial_types), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        start_time = time()
        rate_nbpb = pd.DataFrame(list_rate_RRneuron_dr[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
        # rate_sorted_mean_nbpb, rate_sorted_var_nbpb = compute_mean_var_trial(stm_cnt_dict, rate_nbpb)
        rate_sorted_mean_coll_nbpb, rate_sorted_var_coll_nbpb = compute_mean_var_trial_collapse(stm_cnt_dict, rate_nbpb)

        # Calculate and collect linear slopes for all stimuli
        slopes_nbpb = np.zeros((2, rate_sorted_mean_coll.shape[1]))
        for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):
            # loglog scale
            bool_mean_nb_notzero = rate_sorted_mean_coll_nbpb.loc[:, trial_type] > 0
            popt = np.polyfit(np.log10(rate_sorted_mean_coll_nbpb.loc[bool_mean_nb_notzero, trial_type].values).flatten().astype(np.float32), \
                                np.log10(rate_sorted_var_coll_nbpb.loc[bool_mean_nb_notzero, trial_type].values).flatten().astype(np.float32), 1) # 모든 trial type의 trial 수 같으므로 collapsed 버전 사용
            slopes_nbpb[:, trial_type_ind] = popt.copy()
        list_slopes_all_an_loglog_nbpb[slope_ind] = slopes_nbpb.copy()
        # print(f'sess_ind {sess_ind}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'slopes_katz_indep_sep_RRneuron_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_slopes_all_an_loglog_nbpb_asis', 'list_slopes_all_an_loglog_nbpb'],
                     'list_slopes_all_an_loglog_nbpb_asis': list_slopes_all_an_loglog_nbpb_asis, 'list_slopes_all_an_loglog_nbpb': list_slopes_all_an_loglog_nbpb}, f)
    
    print("Ended Process",c_proc.name)

# %%
# total variance of resampled data
def totvar_resampled(sess_ind, method, n_samples, use_rrep=True, model='negbinom'):

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    num_trial_types = 119
    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    list_target_slopes = [0, 1, 2]
    rng = np.random.default_rng(sess_ind)

    print(f'session index: {sess_ind}')

    rate = list_rate_all[sess_ind]

    # rate_nb loading
    
    # NB+P
    if not use_rrep: # one r for each combination of neuron and stimulus
        method += '_rcomb'
    file_name = 'rate_nb_copula_ABO_sep_' + method + '_' + str(n_samples) + 'samples_realR_all.pickle' # r=inf when var_G=0 (500 samples), r=inf when var<mean (50 samples), r=inf when var<mean (500/50 samples, rcomb)
    with open(file_name, 'rb') as f:
        rate_nb_copula_ABO = pickle.load(f)
        list_rate_nb2_rep = rate_nb_copula_ABO['list_rate_nb2'][sess_ind].copy()

    # # poisson
    # list_rate_nb2_rep = np.full((len(list_target_slopes), rate.shape[0], num_trial_types*n_samples), np.nan)
    # method_poiss = 'rcomb'
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all3.pickle' # lam=mean for comb of neu/stim (500/50 samples)
    # file_name = 'rate_poiss_copula_ABO_sep_' + method_poiss + '_' + str(n_samples) + 'samples_realR_all4.pickle' # lam=var for comb of neu/stim (500/50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_poiss_copula_ABO = pickle.load(f)
    #     list_rate_nb2_rep[0] = rate_poiss_copula_ABO['list_rate_poiss2'][sess_ind].copy()
    # method = method_poiss

    # # katz (nb + poisson + binomial)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all.pickle' # slope 0-2, poiss margin 0.05, binom n>=1 (50 samples)
    # file_name = 'rate_katz_indep_ABO_sep_RRneuron_' + str(n_samples) + 'samples_realR_all2.pickle' # slope 0-2, poiss margin 0.05, binom n>=1, use random_state (50 samples)
    # with open(file_name, 'rb') as f:
    #     rate_nb_copula_ABO = pickle.load(f)
    #     rate = rate_nb_copula_ABO['list_rate_katz_asis'][sess_ind].copy()
    #     list_rate_katz_RRneuron = rate_nb_copula_ABO['list_rate_katz_RRneuron2'][sess_ind].copy()

    rate_sorted = rate.sort_index(axis=1)
    stm = rate_sorted.columns.copy()

    # Multiply by delta t to convert to spike counts (comment this part when using sampled spike count as 'rate')
    rate_sorted = rate_sorted * 0.25

    # Create a counting dictionary for each stimulus
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # trial shuffling
    rate_shuf = pd.DataFrame(np.zeros_like(rate_sorted), columns=rate_sorted.columns)
    for neu_ind in range(rate_sorted.shape[0]):
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            shuf_inds = rng.permutation(rate_sorted.loc[:, trial_type].shape[1])
            rate_shuf.loc[neu_ind, trial_type] = rate_sorted.loc[neu_ind, trial_type].iloc[shuf_inds]
    # rate_sorted = rate_shuf.copy()

    # Compute mean & variance for each stimulus
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], columns=rate_sorted_mean_coll.columns)

    # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
    rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
    rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

    list_totvar_asis = np.nansum(np.log10(rate_sorted_var_coll.values), axis=0)

    list_totvar_RRneuron = np.full((len(list_target_slopes), num_trial_types), np.nan)
    list_totvar_nbp = np.full((len(list_target_slopes), num_trial_types), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
        # if slope_ind in [0, 10, 20]:
        if slope_ind == 1: # 1 for NB+P, 0 for poisson; comment this part for katz
            print(f'target_slope {target_slope:.1f}')

            # # calculate target variance
            # var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
            #                         columns=rate_sorted_var_coll.columns)
            # for trial_type in rate_sorted_var_coll.columns:
            #     var_estim_dr.loc[:, trial_type] = \
            #         np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
            # # var_estim_dr = np.repeat(var_estim_dr, all_stm_counts, axis=1)
            # # print(var_estim_dr)

            # # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed
            # offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr.iloc[1, :])

            # var_rs_noisy = \
            #     pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
            #         .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # collapsed
            # var_rs_noisy = np.repeat(np.array(var_rs_noisy), all_stm_counts, axis=1)

            # # Compute changed residual and add back to the mean            
            # rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            # #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(var_rs_noisy))
            # # print(rate_resid_RRneuron_dr)
            # rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            # rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # convert NaN to 0!

            # # Compute mean and variance of slope-changed data
            # rate_mean_RRneuron_coll, rate_var_RRneuron_coll = \
            #     compute_mean_var_trial_collapse(stm_cnt_dict, rate_RRneuron_dr)

            # # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
            # rate_mean_RRneuron_coll[rate_mean_RRneuron_coll == 0] = np.nan
            # rate_var_RRneuron_coll[rate_var_RRneuron_coll == 0] = np.nan
            # # list_totvar_RRneuron[slope_ind] = gmean(rate_var_RRneuron_coll, axis=0, nan_policy='omit')
            # list_totvar_RRneuron[slope_ind] = np.nansum(np.log10(rate_var_RRneuron_coll.values), axis=0)

            rate_nb = pd.DataFrame(list_rate_nb2_rep[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
            # rate_nb = pd.DataFrame(list_rate_katz_RRneuron[slope_ind], columns=np.repeat(all_stm_unique, n_samples))
            rate_sorted_mean_coll_nb, rate_sorted_var_coll_nb = compute_mean_var_trial_collapse(stm_cnt_dict, rate_nb) # mean, var of sampled pseudo-data
            
            # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
            rate_sorted_mean_coll_nb[rate_sorted_mean_coll_nb == 0] = np.nan
            rate_sorted_var_coll_nb[rate_sorted_var_coll_nb == 0] = np.nan
            # list_totvar_nbp[slope_ind] = gmean(rate_sorted_var_coll_nb, axis=0, nan_policy='omit')
            list_totvar_nbp[slope_ind] = np.nansum(np.log10(rate_sorted_var_coll_nb.values), axis=0)

    # Save into a file
    filename = 'totvar_nb_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    # filename = 'totvar_poiss_sep_' + method + '_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    # filename = 'totvar_katz_indep_sep_RRneuron_' + str(n_samples) + 'samples_realR_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        # pickle.dump({'tree_variables': ['list_totvar_RRneuron', 'list_totvar_nbp'],
        #              'list_totvar_RRneuron': list_totvar_RRneuron, 'list_totvar_nbp': list_totvar_nbp}, f)
        pickle.dump({'tree_variables': ['list_totvar_asis', 'list_totvar_RRneuron', 'list_totvar_nbp'],
                     'list_totvar_asis': list_totvar_asis, 'list_totvar_RRneuron': list_totvar_RRneuron, 'list_totvar_nbp': list_totvar_nbp}, f)
                    
    print("Ended Process",c_proc.name)

# %%
# loading variables

# ABO Neuropixels
with open('resp_matrix_ep_RS_all_32sess_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_RS = dc(resp_matrix_ep_RS_all['list_rate_RS'])
    list_rate_RS_dr = dc(resp_matrix_ep_RS_all['list_rate_RS_dr'])
    list_rate_all = dc(resp_matrix_ep_RS_all['list_rate_all'])
    list_rate_all_dr = dc(resp_matrix_ep_RS_all['list_rate_all_dr'])
    list_slopes_RS_an_loglog = dc(resp_matrix_ep_RS_all['list_slopes_RS_an_loglog'])
    list_slopes_all_att_loglog = dc(resp_matrix_ep_RS_all['list_slopes_all_att_loglog'])
    list_slopes_all_an_loglog = dc(resp_matrix_ep_RS_all['list_slopes_all_an_loglog'])
    list_slopes_all_att_loglog_RRneuron = dc(resp_matrix_ep_RS_all['list_slopes_all_att_loglog_RRneuron'])

    sess_inds_qual_all = dc(resp_matrix_ep_RS_all['sess_inds_qual_all'])

# static gratings
with open('resp_matrix_ep_sg_all_32sess_gpu.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_sg_all = resp_matrix_ep_RS_all['list_rate_sg_all'].copy()
    list_slopes_sg_all_an_loglog = resp_matrix_ep_RS_all['list_slopes_sg_all_an_loglog'].copy()
    list_sg_ori = resp_matrix_ep_RS_all['list_sg_ori'].copy()
    list_sg_sf = resp_matrix_ep_RS_all['list_sg_sf'].copy()
    list_sg_ph = resp_matrix_ep_RS_all['list_sg_ph'].copy()

save_file_name = 'poisson_fit_rp_sep_ABO_all.pickle' # GLM, bfgs, r=inf when var_G=0
save_file_name = 'poisson_fit_rp_sep_ABO_all2.pickle' # GLM, bfgs, r=inf when var<mean
with open(save_file_name, 'rb') as f:
    poisson_fit_rp_ABO_all = pickle.load(f)
    list_r_estim_neu2 = dc(poisson_fit_rp_ABO_all['list_r_estim_neu2']) # katz (nb)
    list_p_estim_neu2 = dc(poisson_fit_rp_ABO_all['list_p_estim_neu2']) # katz (nb, bin)

# %%
# multiprocessing
num_sess = 32

# fit models
mode = 'mean'
if __name__ == '__main__':
    
    with mp.Pool() as pool:
        list_inputs = [[sess_ind, mode] for sess_ind in range(num_sess)]
        
        pool.starmap(fit_discrete_models, list_inputs)

# sample spike counts
method = 'resc_r'
n_samples = 50
model = 'negbinom'
if __name__ == '__main__':
    
    with mp.Pool(processes=16) as pool:
        list_inputs = [[sess_ind, method, n_samples, model] for sess_ind in range(num_sess)] # copula
        pool.starmap(sample_spkcnt_copula, list_inputs)
        
        # list_inputs = [[sess_ind, n_samples, model] for sess_ind in range(num_sess)] # independent
        # pool.starmap(sample_spkcnt_indep, list_inputs)

# decoding
decoder_type = 'SVM'
method = 'resc_r'
n_samples = 50
use_rrep = False
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind, decoder_type, method, n_samples, use_rrep] for sess_ind in range(num_sess)]
        
        pool.starmap(decode_ABO, list_inputs)

# overlap
method = 'resc_r'
n_samples = 50
use_rrep = False
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind, method, n_samples, use_rrep] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_overlap_stimpairs, list_inputs)

# RSA
similarity_type = 'cos_sim'
method = 'resc_r'
n_samples = 50
use_rrep = False
if __name__ == '__main__':
    
    with mp.Pool(processes=8) as pool:
        list_inputs = [[sess_ind, similarity_type, method, n_samples, use_rrep] for sess_ind in range(num_sess)]
        
        pool.starmap(RSA_across_sesspairs_ABO, list_inputs)

# RSA
similarity_type = 'cos_sim'
method = 'resc_r'
n_samples = 50
use_rrep = False
if __name__ == '__main__':
    
    with mp.Pool(processes=8) as pool:
        list_inputs = [[sess_ind, similarity_type, method, n_samples, use_rrep] for sess_ind in range(num_sess)]
        
        pool.starmap(RSA_withinsess_ABO, list_inputs)

# effective dimensionality
method = 'resc_r'
n_samples = 50
use_rrep = False
n_trial_sampling = 100
if __name__ == '__main__':
    
    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind, method, n_samples, use_rrep, n_trial_sampling] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_eff_dim, list_inputs)

# fit slopes and intercepts
n_samples = 50
if __name__ == '__main__':
    
    with mp.Pool(processes=8) as pool:
        list_inputs = [[sess_ind, n_samples] for sess_ind in range(num_sess)]
        
        pool.starmap(linreg_nb, list_inputs)

# total variance
method = 'resc_r'
n_samples = 50
use_rrep = False
model = 'negbinom'
if __name__ == '__main__':
    
    with mp.Pool(processes=8) as pool:
        list_inputs = [[sess_ind, method, n_samples, use_rrep, model] for sess_ind in range(num_sess)]
        
        pool.starmap(totvar_resampled, list_inputs)
