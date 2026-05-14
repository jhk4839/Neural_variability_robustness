# %%
import pickle
from copy import deepcopy as dc
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import sparse
from scipy.stats import wilcoxon, kruskal, mannwhitneyu, sem, spearmanr, rankdata
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from statsmodels.discrete.discrete_model import NegativeBinomial
from itertools import combinations, product, permutations
import seaborn as sns
import math
from time import time
import networkx as nx
import cupy as cp

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

from umap import UMAP

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
def compute_spkcnt_slope_comb(net_params=None, stim_params=None, sim_params=None, misc_params=None):

    save_path = 'D:\\Users\\USER\\Shin Lab\\EI_cluster_network\\'
    save_name_sum = ''
    if net_params is not None:
        for key, value in net_params.items():
            if key != 'baseline_conn_prob':
                save_name_sum += '_' + key + str(value)    
            else:
                save_name_sum += '_' + 'p'
                for p in value.flatten():
                    save_name_sum += str(int(p*10))    
    if stim_params is not None: # if stim_dict was changed from default
        for key, value in stim_params.items():
            # if key in ['n_stim_clusters', 'n_stims', 'stim_amp']:
            # if key in ['n_stim_clusters', 'n_stims', 'n_trials', 'stim_amp']:
                save_name_sum += '_' + key + str(value)
    if sim_params is not None:
        for key, value in sim_params.items():
            save_name_sum += '_' + key + str(value)
    if misc_params is not None:
        for key, value in misc_params.items():
            save_name_sum += '_' + key + str(value)

    save_file_name = 'ei_clust_result' + save_name_sum + '.pickle'
    save_file_name = save_path + save_file_name
    with open(save_file_name, 'rb') as f:
        ei_clust_result = pickle.load(f)
    params_sim = dc(ei_clust_result['_params'])
    
    # spike train
    spiketimes = ei_clust_result['spiketimes'].copy() # (2, num_tot_spikes) -> row 0 is spike times, row 1 is neuron ids
    N_E, N_I = params_sim['N_E'], params_sim['N_I']
    neu_inds_t = spiketimes[1].astype(int).copy()
    neu_inds = np.unique(neu_inds_t)
    spktrn0 = np.zeros((N_E+N_I, int(params_sim['simtime'] + 1)))
    for neu_ind in neu_inds:
        # if neu_ind % 1000 == 0:
        #     print(f'neu_ind {neu_ind}')
        spktrn0[neu_ind, spiketimes[0, neu_inds_t == neu_ind].astype(int)] = 1
    # print((spktrn0 > 0).sum() == spiketimes.shape[1]) # must be equal
    spktrn = spktrn0[:, 1:].copy() # exclude 0 ms

    # spike count mean-var
    n_clusters = net_params['n_clusters']
    n_stims = stim_params['n_stims']; n_trials = stim_params['n_trials']; dur_stim = stim_params['dur_stim']
    if stim_params.get('n_stim_clusters') is not None:
        n_stim_clusters = stim_params['n_stim_clusters']
        if isinstance(n_stim_clusters, str):
            n_stims_temp = n_stims
        else:
            n_stims_temp = np.min([n_stims, math.comb(n_clusters, n_stim_clusters)])
    elif stim_params.get('n_stim_neurons') is not None:
        n_stim_neurons = stim_params['n_stim_neurons']
        n_stims_temp = n_stims

    # spktrn_3d = spktrn.reshape(N_E+N_I, -1, dur_stim) # n_neurons, n_trials, dur_stim
    spktrn_3d = spktrn[:, :int(n_stims_temp*n_trials*dur_stim)].reshape(N_E+N_I, -1, dur_stim) # to cover the case when simtime > total stimulation time
    spkcnt = spktrn_3d.sum(axis=2)
    # print(spkcnt.shape)
    stim_inds_trial = params_sim['stim_inds_trial'].copy()
    spkcnt = pd.DataFrame(spkcnt, columns=stim_inds_trial)
    stims_unique = np.unique(stim_inds_trial)
    cnt_mean, cnt_var = spkcnt.T.groupby(level=0).mean().T, spkcnt.T.groupby(level=0).var().T # (n_neurons, n_stims)
    list_slopes = np.full((2, n_stims), np.nan)
    for stim in stims_unique:
        temp_mean, temp_var = cnt_mean.loc[:, stim].copy(), cnt_var.loc[:, stim].copy()
        bool_mean_notzero = temp_mean > 0
        popt = np.polyfit(np.log10(temp_mean[bool_mean_notzero]), np.log10(temp_var[bool_mean_notzero]), 1)
        list_slopes[:, stim] = popt

    return spkcnt, list_slopes, params_sim

def compute_spkcnt_slope_comb2(net_params=None, stim_params=None, sim_params=None, misc_params=None, across_stimuli=False, spkcnt=None, compute_slope=True):
    
    ''' Same function as 'compute_spkcnt_slope_comb' in .ipynb file as of 2026.5.14. '''

    save_path = 'D:\\Users\\USER\\Shin Lab\\EI_cluster_network\\'
    save_name_sum = ''
    if net_params is not None:
        for key, value in net_params.items():
            if key != 'baseline_conn_prob':
                save_name_sum += '_' + key + str(value)    
            else:
                save_name_sum += '_' + 'p'
                for p in value.flatten():
                    save_name_sum += str(int(p*10))    
    if stim_params is not None: # if stim_dict was changed from default
        for key, value in stim_params.items():
            # if key in ['n_stim_clusters', 'n_stims', 'stim_amp']:
            # if key in ['n_stim_clusters', 'n_stims', 'n_trials', 'stim_amp']:
                save_name_sum += '_' + key + str(value)
    if sim_params is not None:
        for key, value in sim_params.items():
            save_name_sum += '_' + key + str(value)
    if misc_params is not None:
        for key, value in misc_params.items():
            save_name_sum += '_' + key + str(value)

    save_file_name = 'ei_clust_result' + save_name_sum + '.pickle'
    save_file_name = save_path + save_file_name
    with open(save_file_name, 'rb') as f:
        ei_clust_result = pickle.load(f)
    params_sim = dc(ei_clust_result['_params'])
    try:
        adjmat = dc(ei_clust_result['adjmat'])
    except:
        adjmat = None
    
    if spkcnt is None:
        # spike train
        spiketimes = ei_clust_result['spiketimes'].copy() # (2, num_tot_spikes) -> row 0 is spike times, row 1 is neuron ids
        N_E, N_I = params_sim['N_E'], params_sim['N_I']
        neu_inds_t = spiketimes[1].astype(int).copy()
        neu_inds = np.unique(neu_inds_t)
        spktrn0 = np.zeros((N_E+N_I, int(params_sim['simtime'] + 1)))
        for neu_ind in neu_inds:
            # if neu_ind % 1000 == 0:
            #     print(f'neu_ind {neu_ind}')
            spktrn0[neu_ind, spiketimes[0, neu_inds_t == neu_ind].astype(int)] = 1
        # print((spktrn0 > 0).sum() == spiketimes.shape[1]) # must be equal
        spktrn = spktrn0[:, 1:].copy() # exclude 0 ms

        # spike count mean-var
        n_clusters = net_params['n_clusters']
        n_stims = stim_params['n_stims']; n_trials = stim_params['n_trials']; dur_stim = stim_params['dur_stim']
        if stim_params.get('n_stim_clusters') is not None:
            n_stim_clusters = stim_params['n_stim_clusters']
            n_stims_temp = np.min([n_stims, math.comb(n_clusters, n_stim_clusters)])
        elif stim_params.get('n_stim_neurons') is not None:
            n_stim_neurons = stim_params['n_stim_neurons']
            n_stims_temp = n_stims

        # spktrn_3d = spktrn.reshape(N_E+N_I, -1, dur_stim) # n_neurons, n_trials, dur_stim
        spktrn_3d = spktrn[:, :int(n_stims_temp*n_trials*dur_stim)].reshape(N_E+N_I, -1, dur_stim) # to cover the case when simtime > total stimulation time
        spkcnt = spktrn_3d.sum(axis=2)
        # print(spkcnt.shape)
    
    if compute_slope:
        N_E, N_I = params_sim['N_E'], params_sim['N_I']
        n_clusters = net_params['n_clusters']
        n_stims = stim_params['n_stims']; n_trials = stim_params['n_trials']; dur_stim = stim_params['dur_stim']
        if stim_params.get('n_stim_clusters') is not None:
            n_stim_clusters = stim_params['n_stim_clusters']
            n_stims_temp = np.min([n_stims, math.comb(n_clusters, n_stim_clusters)])
        elif stim_params.get('n_stim_neurons') is not None:
            n_stim_neurons = stim_params['n_stim_neurons']
            n_stims_temp = n_stims
        stim_inds_trial = params_sim['stim_inds_trial'].copy()
        spkcnt = pd.DataFrame(spkcnt, columns=stim_inds_trial)
        stims_unique = np.unique(stim_inds_trial)
        cnt_mean, cnt_var = spkcnt.T.groupby(level=0).mean().T, spkcnt.T.groupby(level=0).var().T # n_neurons, n_stims
        list_slopes = np.full((2, n_stims), np.nan) # slope, intercept
        for stim in stims_unique:
            temp_mean, temp_var = cnt_mean.loc[:, stim].copy(), cnt_var.loc[:, stim].copy()
            bool_mean_notzero = temp_mean > 0
            popt = np.polyfit(np.log10(temp_mean.loc[bool_mean_notzero]), np.log10(temp_var.loc[bool_mean_notzero]), 1)
            list_slopes[:, stim] = popt

        if across_stimuli:
            list_slopes_acrtt = np.full((2, N_E+N_I), np.nan)
            for neu_ind in range(N_E+N_I):
                temp_mean, temp_var = cnt_mean.loc[neu_ind].copy(), cnt_var.loc[neu_ind].copy()
                bool_mean_notzero = temp_mean > 0
                # print(bool_mean_notzero.sum())
                if np.any(bool_mean_notzero):
                    popt = np.polyfit(np.log10(temp_mean.loc[bool_mean_notzero]), np.log10(temp_var.loc[bool_mean_notzero]), 1)
                list_slopes_acrtt[:, neu_ind] = popt
            list_slopes = dc([list_slopes_acrtt, list_slopes])
    else:
        list_slopes = None

    return spkcnt, list_slopes, params_sim, adjmat

# %%
# slope and intercept
def linreg_SNN(sess_ind, seed):

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    list_Q = [5, 10, 20, 50, 100]
    n_stims = 5; n_trials = 200; dur_stim = 250; stim_amp = 0.1; stim_overlap = False; match_amp = True; inc_amp = False
    N_E = 1200; N_I = 300
    N = N_E + N_I

    save_file_name = 'RSA_across_within_sess_all27.pickle' # m4, stim_amp 0.1, 10 sessions (seed 1-901), trial/neuron division matching, V_m/conn control
    with open(save_file_name, 'rb') as f:
        RSA_across_within_sess_all = pickle.load(f)
        list_spkcnt_Qstcl = RSA_across_within_sess_all['list_spkcnt_Qstcl'].copy()

    # list_slopes_Qstcl = np.full((len(list_Q), 2, n_stims), np.nan)
    list_slopes_acrneu = np.full((len(list_Q), 2, n_stims), np.nan) # slope, intercept
    list_slopes_acrtt = np.full((len(list_Q), 2, N), np.nan)
    list_params_sim = np.empty(len(list_Q), dtype=object)
    for ind, Q in enumerate(list_Q):
        stcl = int(Q*0.2)
        rep = 'm4'
        print(f'sess_ind={sess_ind}, Q={Q}')
        net_params = dict(zip(['rep', 'n_clusters', 'conn_seed'], [rep, Q, seed])) # Caution on order! Order should reflect the name of loaded file
        stim_params = dict(zip(['n_stim_clusters', 'n_stims', 'n_trials', 'dur_stim', 'stim_amp', 'stim_overlap', 'match_amp', 'inc_amp'],
                               [stcl, n_stims, n_trials, dur_stim, stim_amp, stim_overlap, match_amp, inc_amp]))
        # _, list_slopes_Qstcl[ind], list_params_sim[ind] = compute_spkcnt_slope_comb(net_params=net_params, stim_params=stim_params)
        _, list_slopes, list_params_sim[ind], _ = compute_spkcnt_slope_comb2(net_params=net_params, stim_params=stim_params, across_stimuli=True, spkcnt=list_spkcnt_Qstcl[sess_ind, ind])
        list_slopes_acrtt[ind], list_slopes_acrneu[ind] = list_slopes.copy()

    # Q=1
    similarity_type = 'cos_sim'
    save_file_name = 'RSA_across_within_sess_' + similarity_type + '_all4.pickle' # stim_amp 0.1, 10 sessions (seed 1-901), trial/neuron division matching, V_m/conn control, stim_neu, within sess RSM, Q=1
    with open(save_file_name, 'rb') as f:
        RSA_across_within_sess_all = pickle.load(f)
        list_spkcnt_Qstcl_Q1 = RSA_across_within_sess_all['list_spkcnt_Qstcl'].copy()

    list_Q1 = [1]
    # list_slopes_Qstcl_Q1 = np.full((len(list_Q1), 2, n_stims), np.nan)
    list_slopes_acrneu_Q1 = np.full((len(list_Q1), 2, n_stims), np.nan) # slope, intercept
    list_slopes_acrtt_Q1 = np.full((len(list_Q1), 2, N), np.nan)
    list_params_sim_Q1 = np.empty(len(list_Q1), dtype=object)
    for ind, Q in enumerate(list_Q1):
        stneu = int(N_E*0.2)
        print(f'sess_ind={sess_ind}, Q={Q}')
        net_params = dict(zip(['n_clusters', 'conn_seed'], [Q, seed])) # Caution on order! Order should reflect the name of loaded file
        stim_params = dict(zip(['n_stim_neurons', 'n_stims', 'n_trials', 'dur_stim', 'stim_amp', 'stim_overlap', 'match_amp', 'inc_amp'],
                               [stneu, n_stims, n_trials, dur_stim, stim_amp, stim_overlap, match_amp, inc_amp]))
        # _, list_slopes_Qstcl_Q1[ind], list_params_sim_Q1[ind] = compute_spkcnt_slope_comb(net_params=net_params, stim_params=stim_params)
        _, list_slopes, list_params_sim_Q1[ind], _ = compute_spkcnt_slope_comb2(net_params=net_params, stim_params=stim_params, across_stimuli=True, spkcnt=list_spkcnt_Qstcl_Q1[sess_ind, ind])
        list_slopes_acrtt_Q1[ind], list_slopes_acrneu_Q1[ind] = list_slopes.copy()

    # Save into a file
    filename = 'slopes_SNN_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        # pickle.dump({'tree_variables': ['list_slopes_Qstcl', 'list_slopes_Qstcl_Q1', 'list_params_sim', 'list_params_sim_Q1'],
        #              'list_slopes_Qstcl': list_slopes_Qstcl, 'list_slopes_Qstcl_Q1': list_slopes_Qstcl_Q1,
        #              'list_params_sim': list_params_sim, 'list_params_sim_Q1': list_params_sim_Q1}, f)
        pickle.dump({'tree_variables': ['list_slopes_acrneu', 'list_slopes_acrtt', 'list_slopes_acrneu_Q1', 'list_slopes_acrtt_Q1', 'list_params_sim', 'list_params_sim_Q1'],
                     'list_slopes_acrneu': list_slopes_acrneu, 'list_slopes_acrtt': list_slopes_acrtt, 'list_slopes_acrneu_Q1': list_slopes_acrneu_Q1, 'list_slopes_acrtt_Q1': list_slopes_acrtt_Q1,
                     'list_params_sim': list_params_sim, 'list_params_sim_Q1': list_params_sim_Q1}, f)
            
    print("Ended Process",c_proc.name)

# %%
# decoding stimuli
def decode_SNN(sess_ind, seed, decoder_type):

    ''' decoder_type is SVM, logit, RF, kNN '''

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    rng = np.random.default_rng(seed)
    list_Q = [5, 10, 20, 50, 100]
    # list_Q = [1]
    n_stims = 5; n_trials = 200; dur_stim = 250; stim_amp = 0.1; stim_overlap = False; match_amp = True
    N_E = 1200
    
    list_spkcnt_Qstcl = np.empty(len(list_Q), dtype=object)
    list_slopes_Qstcl = np.full((len(list_Q), n_stims), np.nan)
    list_params_sim = np.empty(len(list_Q), dtype=object)
    list_mean_confusion_test = np.full((len(list_Q), n_stims, n_stims), np.nan)
    list_mean_accuracy = np.full(len(list_Q), np.nan)
    for ind, Q in enumerate(list_Q):
        stcl = int(Q*0.2)
        stneu = int(N_E*0.2) # Q = 1; replace the parameter 'n_stim_clusters' and 'stcl' with 'n_stim_neurons' and 'stneu'
        rep = 'm4' # remove this parameter when Q = 1
        print(f'sess_ind={sess_ind}, Q={Q}')
        net_params = dict(zip(['rep', 'n_clusters', 'conn_seed'], [rep, Q, seed])) # Caution on order! Order should reflect the name of loaded file
        stim_params = dict(zip(['n_stim_clusters', 'n_stims', 'n_trials', 'dur_stim', 'stim_amp', 'stim_overlap', 'match_amp'],
                               [stcl, n_stims, n_trials, dur_stim, stim_amp, stim_overlap, match_amp]))
        list_spkcnt_Qstcl[ind], list_slopes_Qstcl[ind], list_params_sim[ind] = compute_spkcnt_slope_comb(net_params=net_params, stim_params=stim_params)
        rate = list_spkcnt_Qstcl[ind].copy()
        params_sim = dc(list_params_sim[ind])

        # num_trial_types = np.min([n_stims, math.comb(Q, stcl)])
        num_trial_types = n_stims
        num_trials = n_trials
        n_splits = 10

        train_stimuli = range(num_trial_types)

        rate_sorted = rate.sort_index(axis=1)
        stm = rate_sorted.columns.copy()

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

        # Compute mean & variance for each stimulus
        rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
        rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

        # decoding cross-validation
        kfold = KFold(n_splits=n_splits)
        stkfold = StratifiedKFold(n_splits=n_splits)

        # Re-convert to 2D response matrix
        label_train = stm.copy()
        rate_train = rate_sorted.copy()

        list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
        list_accuracy = np.full(n_splits, np.nan)
        start_time = time()
        for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
            X_train, X_test = rate_train.T.iloc[train_index].copy(), rate_train.T.iloc[test_index].copy() # train, test data/label
            y_train, y_test = label_train[train_index].copy(), label_train[test_index].copy()

            mean_ = X_train.mean(axis=0)
            X_train = X_train.sub(mean_, axis=1) # train data mean centering
            X_test = X_test.sub(mean_, axis=1)

            if decoder_type == 'SVM':
                clf = svm.SVC(kernel='linear')
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

        print(f'sess_ind={sess_ind}, Q={Q}, duration {(time()-start_time)/60:.2f} min')

        # calculate cross-validation average test confusion matrix/test accuracy
        mean_confusion_test = sum(list_confusion_test) / n_splits
        mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
        # print(mean_confusion_test.round(3))
        list_mean_confusion_test[ind] = mean_confusion_test.copy()

        mean_accuracy = np.mean(list_accuracy)
        # print(round(mean_accuracy, ndigits=3))
        list_mean_accuracy[ind] = mean_accuracy

    # Save into a file
    filename = decoder_type + '_decoding_SNN_allstim_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_spkcnt_Qstcl', 'list_slopes_Qstcl', 'list_params_sim', 'list_mean_confusion_test', 'list_mean_accuracy'],
                     'list_spkcnt_Qstcl': list_spkcnt_Qstcl, 'list_slopes_Qstcl': list_slopes_Qstcl, 'list_params_sim': list_params_sim,
                     'list_mean_confusion_test': list_mean_confusion_test, 'list_mean_accuracy': list_mean_accuracy}, f)
    
    print("Ended Process",c_proc.name)

# %%
# RSM for a session (for RSA across session pairs) & RSA within sessions
def RSA_across_within_sess(sess_ind, seed, similarity_type='cos_sim'):

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    rng = np.random.default_rng(seed)
    list_Q = [5, 10, 20, 50, 100]
    # list_Q = [1]
    n_stims = 5; n_trials = 200; dur_stim = 250; stim_amp = 0.1; stim_overlap = False; match_amp = True
    n_neu_sampling = 10
    N_E = 1200

    list_spkcnt_Qstcl = np.empty(len(list_Q), dtype=object)
    list_slopes_Qstcl = np.full((len(list_Q), 2, n_stims), np.nan) # slope, intercept
    list_params_sim = np.empty(len(list_Q), dtype=object)
    list_RSM_mean = np.full((len(list_Q), n_stims, n_stims), np.nan)
    list_corr_withinsess = np.full((len(list_Q), n_neu_sampling, 3), np.nan)
    
    rand_trial_inds = rng.permutation(range(n_trials))
    num_neurons = 1500
    list_neu_div_inds = np.zeros((n_neu_sampling, 2, num_neurons//2), dtype=int)
    for neu_sample_ind in range(n_neu_sampling):
        neu_inds_permuted = rng.permutation(range(num_neurons))
        neu_div_inds1 = neu_inds_permuted[:int(num_neurons/2)].copy() # 5:5 partitioning
        neu_div_inds2 = neu_inds_permuted[int(num_neurons/2):].copy()
        if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # if num_neurons is odd number
            neu_div_inds2 = neu_div_inds2[:-1].copy()
        list_neu_div_inds[neu_sample_ind] = dc([neu_div_inds1, neu_div_inds2])

    # Iterate over Q
    list_RSM_neu1_all = np.full((len(list_Q), n_neu_sampling, n_stims, n_stims), np.nan)
    list_RSM_neu2_all = np.full((len(list_Q), n_neu_sampling, n_stims, n_stims), np.nan)
    for ind, Q in enumerate(list_Q):
        stcl = int(Q*0.2)
        stneu = int(N_E*0.2) # Q = 1; replace the parameter 'n_stim_clusters' and 'stcl' with 'n_stim_neurons' and 'stneu'
        rep = 'm4' # remove this parameter when Q = 1
        print(f'sess_ind={sess_ind}, Q={Q}')
        net_params = dict(zip(['rep', 'n_clusters', 'conn_seed'], [rep, Q, seed])) # Caution on order! Order should reflect the name of loaded file
        stim_params = dict(zip(['n_stim_clusters', 'n_stims', 'n_trials', 'dur_stim', 'stim_amp', 'stim_overlap', 'match_amp'],
                               [stcl, n_stims, n_trials, dur_stim, stim_amp, stim_overlap, match_amp]))
        list_spkcnt_Qstcl[ind], list_slopes_Qstcl[ind], list_params_sim[ind] = compute_spkcnt_slope_comb(net_params=net_params, stim_params=stim_params)
        rate = list_spkcnt_Qstcl[ind].copy()
        params_sim = dc(list_params_sim[ind])

        # num_trial_types = np.min([n_stims, math.comb(Q, stcl)])
        num_trial_types = n_stims
        num_trials = n_trials
        num_neurons = rate.shape[0]

        stm = rate.columns.copy()
        rate_sorted = rate.sort_index(axis=1)
        rate_sorted = np.reshape(rate_sorted.values, (num_neurons, num_trial_types, num_trials))

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True)
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], num_trials, axis=2)

        # trial order re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, rand_trial_inds]

        # repeat calculating similarity matrices

        # n_neurons x n_stimuli 2D matrix sampling
        tt_pairs = list(combinations(range(num_trials), 2))
        n_sampling = np.min([len(tt_pairs), 10000])
        # n_sampling = len(tt_pairs)
        
        list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))

        count = 0
        for sampling_ind in range(n_sampling):
            rate_sampled_trials1 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][0]]).copy()
            rate_sampled_trials2 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][1]]).copy()

            RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2))
            list_RSM[sampling_ind] = RSM.copy()

            count += 1
            if count % 5000 == 0:
                print(f'count: {count}')

        RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
        list_RSM_mean[ind, :num_trial_types, :num_trial_types] = RSM_mean.copy()

        # Iterate over neuron partitionings
        start_time = time()
        for neu_sample_ind in range(n_neu_sampling):
            # print(f'sess_ind={sess_ind}, Q={Q}, neu_sample_ind = {neu_sample_ind}')
            
            # Partition neurons
            neu_div_inds1, neu_div_inds2 = list_neu_div_inds[neu_sample_ind].copy()
            rate_sorted1 = rate_sorted[neu_div_inds1].copy()
            rate_sorted2 = rate_sorted[neu_div_inds2].copy()

            # repeat calculating similarity matrices
            
            # n_neurons x n_stimuli 2D matrix sampling
            
            tt_pairs = list(combinations(range(num_trials), 2))
            # random.shuffle(tt_pairs)
            n_sampling = np.min([len(tt_pairs), 10000])
            # n_sampling = len(tt_pairs)

            list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
            list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

            count = 0
            for sampling_ind in range(n_sampling):
                rate_sampled_trials1_1 = np.squeeze(rate_sorted1[:, :, tt_pairs[sampling_ind][0]]).copy()
                rate_sampled_trials1_2 = np.squeeze(rate_sorted1[:, :, tt_pairs[sampling_ind][1]]).copy()
                rate_sampled_trials2_1 = np.squeeze(rate_sorted2[:, :, tt_pairs[sampling_ind][0]]).copy()
                rate_sampled_trials2_2 = np.squeeze(rate_sorted2[:, :, tt_pairs[sampling_ind][1]]).copy()

                RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2))
                RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2))

                # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos))
                list_RSM_neu1[sampling_ind] = RSM1.copy()
                list_RSM_neu2[sampling_ind] = RSM2.copy()

                count += 1
                if count % 5000 == 0:
                    print(f'count: {count}')

            RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0)
            RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
            list_RSM_neu1_all[ind, neu_sample_ind], list_RSM_neu2_all[ind, neu_sample_ind] = RSM_mean_neu1.copy(), RSM_mean_neu2.copy()

            # exclude diagonal
            RSM_mean_neu1[np.diag_indices(num_trial_types)] = np.nan
            RSM_mean_neu2[np.diag_indices(num_trial_types)] = np.nan

            list_corr_withinsess[ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_withinsess[ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_withinsess[ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
        print(f'sess_ind={sess_ind}, Q={Q}, duration {(time()-start_time)/60:.2f} min')

    # Save into a file
    filename = 'RSA_across_within_sess_' + similarity_type + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_spkcnt_Qstcl', 'list_slopes_Qstcl', 'list_params_sim', 'list_RSM_mean', 'list_RSM_neu1_all', 'list_RSM_neu2_all', 'list_corr_withinsess'],
                     'list_spkcnt_Qstcl': list_spkcnt_Qstcl, 'list_slopes_Qstcl': list_slopes_Qstcl, 'list_params_sim': list_params_sim,
                     'list_RSM_mean': list_RSM_mean, 'list_RSM_neu1_all': list_RSM_neu1_all, 'list_RSM_neu2_all': list_RSM_neu2_all, 'list_corr_withinsess': list_corr_withinsess}, f)
    
    print("Ended Process",c_proc.name)

# %%
# Effective dimensionality
def compute_eff_dim(sess_ind, seed, n_trial_sampling=100):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    rng = np.random.default_rng(seed)
    list_Q = [5, 10, 20, 50, 100]
    # list_Q = [1]
    n_stims = 5; n_trials = 200; dur_stim = 250; stim_amp = 0.1; stim_overlap = False; match_amp = True
    N_E = 1200

    list_spkcnt_Qstcl = np.empty(len(list_Q), dtype=object)
    list_slopes_Qstcl = np.full((len(list_Q), 2, n_stims), np.nan) # slope, intercept
    list_params_sim = np.empty(len(list_Q), dtype=object)

    list_dim = np.zeros((len(list_Q), n_stims))
    list_dim_global = np.zeros((len(list_Q), 2))
    list_dim_sam = np.zeros((len(list_Q), n_trial_sampling))

    n_tot_trials = n_stims * n_trials
    rand_tt_inds = rng.permutation(range(n_tot_trials))

    # Iterate over Q
    for ind, Q in enumerate(list_Q):
        stcl = int(Q*0.2)
        stneu = int(N_E*0.2) # Q = 1; replace the parameter 'n_stim_clusters' and 'stcl' with 'n_stim_neurons' and 'stneu'
        rep = 'm4' # remove this parameter when Q = 1
        print(f'sess_ind={sess_ind}, Q={Q}')

        net_params = dict(zip(['rep', 'n_clusters', 'conn_seed'], [rep, Q, seed])) # Caution on order! Order should reflect the name of loaded file
        stim_params = dict(zip(['n_stim_clusters', 'n_stims', 'n_trials', 'dur_stim', 'stim_amp', 'stim_overlap', 'match_amp'],
                               [stcl, n_stims, n_trials, dur_stim, stim_amp, stim_overlap, match_amp]))
        list_spkcnt_Qstcl[ind], list_slopes_Qstcl[ind], list_params_sim[ind] = compute_spkcnt_slope_comb(net_params=net_params, stim_params=stim_params)
        rate = list_spkcnt_Qstcl[ind].copy()
        params_sim = dc(list_params_sim[ind])

        # num_trial_types = np.min([n_stims, math.comb(Q, stcl)])
        num_trial_types = n_stims
        num_trials = n_trials
        num_neurons = rate.shape[0]
   
        rate_sorted = rate.sort_index(axis=1)
        stm = rate_sorted.columns.copy()

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
            list_dim[ind, trial_type_ind] = (np.trace(cf)**2 / np.sum(cf * cf)) / rate_sorted.shape[0]
        cf_all = np.cov(rate_sorted)
        list_dim_global[ind, 0] = ((np.trace(cf_all)**2) / np.sum(cf_all * cf_all)) / rate_sorted.shape[0]
        cf_cen = np.cov(rate_sorted_mean_coll)
        list_dim_global[ind, 1] = ((np.trace(cf_cen)**2) / np.sum(cf_cen * cf_cen)) / rate_sorted.shape[0]

        rate = rate_sorted.iloc[:, rand_tt_inds].copy()
        for t_sam_ind in range(n_trial_sampling):
            rate_sam = np.full_like(rate_sorted_mean_coll, np.nan)
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                rate_tt = rate.loc[:, trial_type].copy()
                rate_sam[:, trial_type_ind] = rate_tt.iloc[:, rng.choice(range(rate_tt.shape[1]), 1)[0]].copy()
            cf_sam = np.cov(rate_sam)
            list_dim_sam[ind, t_sam_ind] = ((np.trace(cf_sam)**2) / np.sum(cf_sam * cf_sam)) / rate_sorted.shape[0]

    # Save into a file
    filename = 'eff_dim_DC_SNN_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_spkcnt_Qstcl', 'list_slopes_Qstcl', 'list_params_sim', 'list_dim', 'list_dim_global', 'list_dim_sam'],
                    'list_spkcnt_Qstcl': list_spkcnt_Qstcl, 'list_slopes_Qstcl': list_slopes_Qstcl, 'list_params_sim': list_params_sim,
                    'list_dim': list_dim, 'list_dim_global': list_dim_global, 'list_dim_sam': list_dim_sam}, f)

    print("Ended Process", c_proc.name)

# %%
# multiprocessing

# fit slopes and intercepts
n_sessions = 10
list_seed = np.linspace(1, 901, n_sessions, endpoint=True).astype(int)
if __name__ == '__main__':
    
    with mp.Pool(processes=5) as pool:
        list_inputs = [[sess_ind, seed] for sess_ind, seed in enumerate(list_seed)]
        
        pool.starmap(linreg_SNN, list_inputs)

# decoding
n_sessions = 10
list_seed = np.linspace(1, 901, n_sessions, endpoint=True).astype(int)
decoder_type = 'SVM'
if __name__ == '__main__':
    
    with mp.Pool(processes=10) as pool:
        list_inputs = [[sess_ind, seed, decoder_type] for sess_ind, seed in enumerate(list_seed)]
        
        pool.starmap(decode_SNN, list_inputs)

# RSA across/within session
n_sessions = 10
list_seed = np.linspace(1, 901, n_sessions, endpoint=True).astype(int)
similarity_type = 'cos_sim'
if __name__ == '__main__':
    
    with mp.Pool(processes=10) as pool:
        list_inputs = [[sess_ind, seed, similarity_type] for sess_ind, seed in enumerate(list_seed)]
        
        pool.starmap(RSA_across_within_sess, list_inputs)

# effective dimensionality
n_sessions = 10
list_seed = np.linspace(1, 901, n_sessions, endpoint=True).astype(int)
n_trial_sampling = 100
if __name__ == '__main__':

    with mp.Pool() as pool:
        list_inputs = [[sess_ind, seed, n_trial_sampling] for sess_ind, seed in enumerate(list_seed)]
        
        pool.starmap(compute_eff_dim, list_inputs)
