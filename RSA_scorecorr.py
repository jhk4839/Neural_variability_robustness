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
# cosine similarity 계산 함수
def cos_sim(x, y):
    # x, y 각각 1d vector

    # NaN 제거
    x, y = np.array(x), np.array(y)
    bool_notnan = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[bool_notnan].copy(), y[bool_notnan].copy()

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# %%
# column normalization 함수
def normc(matrix):
    '''matrix는 2D'''

    # if isinstance(matrix, np.ndarray):
    #     matrix_normalized = np.zeros_like(matrix)
    #     for column_ind in range(matrix.shape[1]):
    #         matrix_normalized[:, column_ind] = matrix[:, column_ind] / np.linalg.norm(matrix[:, column_ind])

    # elif isinstance(matrix, pd.DataFrame):
    #     matrix_normalized = pd.DataFrame(np.zeros_like(matrix), columns=matrix.columns, index=matrix.index)
    #     for column_ind, column_name in enumerate(matrix.columns):
    #         matrix_normalized.iloc[:, column_ind] = matrix.iloc[:, column_ind] / np.linalg.norm(matrix.iloc[:, column_ind])

    matrix_normalized = matrix / np.linalg.norm(matrix, axis=0)

    return matrix_normalized

# %%
# RSA across session pairs (ABO Neuropixels, RRneuron)

def RSA_across_sesspairs_ABO(slope_ind, target_slope, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # 각 trial type에서 trial sampling할 횟수
    num_trials = 50
    num_trial_types = 119
    num_sess = 32

    print(f'target slope {target_slope:.1f}')

    # 모든 session에 대해 iteration

    np.random.seed(0) # as-is와 RRneuron의 trial 순서 & shuffling을 match하기 위해.

    list_RSM_mean_asis = np.zeros((num_sess, num_trial_types, num_trial_types))
    list_RSM_mean_RRneuron = np.zeros((num_sess, num_trial_types, num_trial_types))
    list_rate_RRneuron_dr = np.empty(num_sess, dtype=object)

    # list_sess_inds = np.delete(np.arange(num_sess).astype(int), [0, 6])
    for ind in range(num_sess):

        print(f'ind: {ind}')

        rate = list_rate_RS[ind].copy()
        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # delta t 곱해서 spike count로 만들기
        rate = rate * 0.25

        # stm type별 counting dictionary 제작
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # 3D matrix로 만들기
        min_num_trials = np.min(all_stm_counts) # session 4는 trial repeat 수가 균질하지 않으므로 최솟값만 취함 (참고: 47개)

        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
            list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        rate = np.stack(list_rate_tt, axis=2)
        rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
    
        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
        
        list_slopes_dr = list_slopes_RS_an_loglog[ind].copy()

        # # trial shuffling
        # rate_shuf = np.zeros_like(rate_sorted)
        # for neu_ind in range(rate_sorted.shape[0]):
        #     shuf_inds = np.random.permutation(rate_sorted.shape[2])
        #     rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() # transpose 주의!
        # rate_sorted = rate_shuf.copy()

        # trial 순서 re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        if slope_ind == 0:

            # RSM 반복 제작

            if similarity_type == 'geodesic' or similarity_type == 'isomap':
                rate_sorted_2d = np.reshape(rate_sorted, (-1, num_trial_types*min_num_trials))

                # isomap
                n_components = 1 # 목표 차원 수
                # n_components = rate_sorted.shape[0] # 목표 차원 수
                n_neighbors = 5 # 이웃 점 개수

                isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
                
                rate_sorted_2d_isomap = isomap.fit_transform(rate_sorted_2d.T).T
                dist_matrix_asis = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

                if similarity_type == 'isomap':
                    dist_matrix_asis = cdist(rate_sorted_2d_isomap.T, rate_sorted_2d_isomap.T, 'euclidean')

            # n_neurons x n_stimuli 2D matrix sampling

            # if ind == 3:
            #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
            tt_pairs = list(combinations(range(min_num_trials), 2))
            n_sampling = np.min([len(tt_pairs), 10000])
            # n_sampling = len(tt_pairs)

            list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
            
            count = 0
            for sampling_ind in range(n_sampling):
                
                if similarity_type == 'cos_sim':
                    rate_sampled_trials1 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                    rate_sampled_trials2 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][1]]).copy()

                    # RSM 제작
                    RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                    # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) # lower triangle도 대칭으로 채우기
                    list_RSM[sampling_ind] = RSM.copy()

                else:
                    row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                    col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                    RSM = dist_matrix_asis[row_inds, :][:, col_inds].copy()
                    list_RSM[sampling_ind] = RSM.copy()
                
                count += 1
                if count % 100 == 0:
                    print(f'count: {count}')

            RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
            list_RSM_mean_asis[ind] = RSM_mean.copy()

        # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        # RRneuron

        # RRneuron var 계산
        var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
        offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

        var_rs_noisy = \
            pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
        var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

        # rate residual RR 계산 & mean과 다시 합하기            
        rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
            * np.sqrt(var_rs_noisy)
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # NaN을 다시 0으로 바꾸기!

        list_rate_RRneuron_dr[ind] = rate_RRneuron_dr.copy()

        # # trial 순서 re-randomization
        # for trial_type_ind in range(num_trial_types):
        #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        # RSM 반복 제작

        if similarity_type == 'geodesic' or similarity_type == 'isomap':
            rate_RRneuron_dr_2d = np.reshape(rate_RRneuron_dr, (-1, num_trial_types*min_num_trials))

            # isomap
            n_components = 1 # 목표 차원 수
            # n_components = rate_sorted.shape[0] # 목표 차원 수
            n_neighbors = 5 # 이웃 점 개수

            isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
            
            rate_RRneuron_dr_2d_isomap = isomap.fit_transform(rate_RRneuron_dr_2d.T).T
            dist_matrix = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

            if similarity_type == 'isomap':
                dist_matrix = cdist(rate_RRneuron_dr_2d_isomap.T, rate_RRneuron_dr_2d_isomap.T, 'euclidean')

        # n_neurons x n_stimuli 2D matrix sampling

        # if ind == 3:
        #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
        tt_pairs = list(combinations(range(min_num_trials), 2))
        n_sampling = np.min([len(tt_pairs), 10000])
        # n_sampling = len(tt_pairs)

        list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
        
        count = 0
        for sampling_ind in range(n_sampling):
            
            if similarity_type == 'cos_sim':
                rate_sampled_trials1 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                rate_sampled_trials2 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][1]]).copy()

                RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) # lower triangle도 대칭으로 채우기
                list_RSM[sampling_ind] = RSM.copy()

            else:
                row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                RSM = dist_matrix[row_inds, :][:, col_inds].copy()
                list_RSM[sampling_ind] = RSM.copy()
            
            count += 1
            if count % 100 == 0:
                print(f'count: {count}')

        RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
        list_RSM_mean_RRneuron[ind] = RSM_mean.copy()

    # Spearman correlation across session pairs
    sess_pairs = list(combinations(range(num_sess), 2))
    list_corr_sesspair_asis = np.zeros((len(sess_pairs), 3)) # correlation 측정법 3가지
    list_corr_sesspair = np.zeros((len(sess_pairs), 3)) # correlation 측정법 3가지
    for pair_ind, pair in enumerate(sess_pairs):
        if slope_ind == 0:
            RSM_mean_neu1 = list_RSM_mean_asis[pair[0]].copy()
            RSM_mean_neu2 = list_RSM_mean_asis[pair[1]].copy()

            list_corr_sesspair_asis[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_sesspair_asis[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_sesspair_asis[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
        
        RSM_mean_neu1 = list_RSM_mean_RRneuron[pair[0]].copy()
        RSM_mean_neu2 = list_RSM_mean_RRneuron[pair[1]].copy()

        list_corr_sesspair[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
        bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
        list_corr_sesspair[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
        list_corr_sesspair[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # 파일에 저장
    filename = 'RSM_corr_sesspair_ABO_RSneu_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis', 'list_corr_sesspair_asis', 'list_rate_RRneuron_dr', 'list_RSM_mean_RRneuron', 'list_corr_sesspair'], \
                     'list_RSM_mean_asis': list_RSM_mean_asis, 'list_corr_sesspair_asis': list_corr_sesspair_asis,
                     'list_rate_RRneuron_dr': list_rate_RRneuron_dr, 'list_RSM_mean_RRneuron': list_RSM_mean_RRneuron, 'list_corr_sesspair': list_corr_sesspair}, f)

    print("Ended Process", c_proc.name)

# %%
# %%
# RSA across session pairs (ABO Neuropixels, RRneuron)

def RSA_across_sesspairs_ABO_HVA(slope_ind, target_slope, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # 각 trial type에서 trial sampling할 횟수
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
        
        # 모든 session에 대해 iteration

        list_RSM_mean_asis = np.full((num_sess, num_trial_types, num_trial_types), np.nan)
        list_RSM_mean_RRneuron = np.full((num_sess, num_trial_types, num_trial_types), np.nan)
        list_rate_RRneuron_dr = np.empty(num_sess, dtype=object)

        # list_sess_inds = np.delete(np.arange(num_sess).astype(int), [0, 6])
        for ind in range(num_sess):

            print(f'area: {area}, ind: {ind}')

            rate = list_rate_all_HVA[area][ind].copy()
            if np.any(rate) == True: # neuron이 있는 경우

                # rate_sorted = rate.sort_index(axis=1)
                stm = rate.columns.copy()

                # delta t 곱해서 spike count로 만들기
                rate = rate * 0.25

                # stm type별 counting dictionary 제작
                all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
                stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
                
                # 3D matrix로 만들기
                min_num_trials = np.min(all_stm_counts) # session 4는 trial repeat 수가 균질하지 않으므로 최솟값만 취함 (참고: 47개)

                list_rate_tt = [None] * num_trial_types
                for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
                    list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

                rate = np.stack(list_rate_tt, axis=2)
                rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
            
                rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
                rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
                    np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
                
                list_slopes_dr = list_slopes_all_an_loglog_HVA[area][ind].copy()

                # trial 순서 re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                if slope_ind == 0:

                    # RSM 반복 제작

                    if similarity_type == 'geodesic' or similarity_type == 'isomap':
                        rate_sorted_2d = np.reshape(rate_sorted, (-1, num_trial_types*min_num_trials))

                        # isomap
                        n_components = 1 # 목표 차원 수
                        # n_components = rate_sorted.shape[0] # 목표 차원 수
                        n_neighbors = 5 # 이웃 점 개수

                        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
                        
                        rate_sorted_2d_isomap = isomap.fit_transform(rate_sorted_2d.T).T
                        dist_matrix_asis = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

                        if similarity_type == 'isomap':
                            dist_matrix_asis = cdist(rate_sorted_2d_isomap.T, rate_sorted_2d_isomap.T, 'euclidean')

                    # n_neurons x n_stimuli 2D matrix sampling

                    # if ind == 3:
                    #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
                    tt_pairs = list(combinations(range(min_num_trials), 2))
                    n_sampling = np.min([len(tt_pairs), 10000])
                    # n_sampling = len(tt_pairs)

                    list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
                    
                    count = 0
                    for sampling_ind in range(n_sampling):
                        
                        if similarity_type == 'cos_sim':
                            rate_sampled_trials1 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                            rate_sampled_trials2 = np.squeeze(rate_sorted[:, :, tt_pairs[sampling_ind][1]]).copy()

                            # RSM 제작
                            RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                            # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) # lower triangle도 대칭으로 채우기
                            list_RSM[sampling_ind] = RSM.copy()

                        else:
                            row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                            col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                            RSM = dist_matrix_asis[row_inds, :][:, col_inds].copy()
                            list_RSM[sampling_ind] = RSM.copy()
                        
                        count += 1
                        if count % 1000 == 0:
                            print(f'count: {count}')

                    RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
                    list_RSM_mean_asis[ind] = RSM_mean.copy()

                # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

                # RRneuron var 계산
                var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

                # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
                # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
                offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

                var_rs_noisy = \
                    pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                        / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
                var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

                # rate residual RR 계산 & mean과 다시 합하기            
                rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
                # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                    * np.sqrt(var_rs_noisy)
                # print(rate_resid_RRneuron_dr)
                rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
                rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # NaN을 다시 0으로 바꾸기!      
                list_rate_RRneuron_dr[ind] = rate_RRneuron_dr.copy()

                # # trial 순서 re-randomization
                # for trial_type_ind in range(num_trial_types):
                #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                # RSM 반복 제작

                if similarity_type == 'geodesic' or similarity_type == 'isomap':
                    rate_RRneuron_dr_2d = np.reshape(rate_RRneuron_dr, (-1, num_trial_types*min_num_trials))

                    # isomap
                    n_components = 1 # 목표 차원 수
                    # n_components = rate_sorted.shape[0] # 목표 차원 수
                    n_neighbors = 5 # 이웃 점 개수

                    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
                    
                    rate_RRneuron_dr_2d_isomap = isomap.fit_transform(rate_RRneuron_dr_2d.T).T
                    dist_matrix = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

                    if similarity_type == 'isomap':
                        dist_matrix = cdist(rate_RRneuron_dr_2d_isomap.T, rate_RRneuron_dr_2d_isomap.T, 'euclidean')

                # n_neurons x n_stimuli 2D matrix sampling

                # if ind == 3:
                #     list_num_trials = [rate_RRneuron_dr.loc[:, trial_type].shape[1] for trial_type in rate_sorted_mean_coll.columns] 
                tt_pairs = list(combinations(range(min_num_trials), 2))
                n_sampling = np.min([len(tt_pairs), 10000])
                # n_sampling = len(tt_pairs)

                list_RSM = np.zeros((n_sampling, num_trial_types, num_trial_types))
                
                count = 0
                for sampling_ind in range(n_sampling):
                    
                    if similarity_type == 'cos_sim':
                        rate_sampled_trials1 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                        rate_sampled_trials2 = np.squeeze(rate_RRneuron_dr[:, :, tt_pairs[sampling_ind][1]]).copy()

                        # RSM 제작
                        RSM = np.array(normc(rate_sampled_trials1).T) @ np.array(normc(rate_sampled_trials2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                        # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) # lower triangle도 대칭으로 채우기
                        list_RSM[sampling_ind] = RSM.copy()

                    else:
                        row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                        col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                        RSM = dist_matrix[row_inds, :][:, col_inds].copy()
                        list_RSM[sampling_ind] = RSM.copy()
                    
                    count += 1
                    if count % 1000 == 0:
                        print(f'count: {count}')

                RSM_mean = np.nanmean(list_RSM, axis=0) # nanmean!
                list_RSM_mean_RRneuron[ind] = RSM_mean.copy()

        # Spearman correlation across session pairs
        sess_pairs = list(combinations(range(num_sess), 2))
        list_corr_sesspair_asis = np.zeros((len(sess_pairs), 3)) # correlation 측정법 3가지
        list_corr_sesspair = np.zeros((len(sess_pairs), 3)) # correlation 측정법 3가지
        for pair_ind, pair in enumerate(sess_pairs):
            if slope_ind == 0:
                RSM_mean_neu1 = list_RSM_mean_asis[pair[0]].copy()
                RSM_mean_neu2 = list_RSM_mean_asis[pair[1]].copy()

                list_corr_sesspair_asis[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_sesspair_asis[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_sesspair_asis[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())                
            
            RSM_mean_neu1 = list_RSM_mean_RRneuron[pair[0]].copy()
            RSM_mean_neu2 = list_RSM_mean_RRneuron[pair[1]].copy()

            list_corr_sesspair[pair_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_sesspair[pair_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_sesspair[pair_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

        list_RSM_mean_asis_HVA[area] = list_RSM_mean_asis.copy()
        list_corr_sesspair_asis_HVA[area] = list_corr_sesspair_asis.copy()

        list_RSM_mean_RRneuron_HVA[area] = list_RSM_mean_RRneuron.copy()
        list_rate_RRneuron_dr_HVA[area] = list_rate_RRneuron_dr.copy()
        list_corr_sesspair_HVA[area] = list_corr_sesspair.copy()
    
    # 파일에 저장
    filename = 'RSM_corr_sesspair_ABO_HVA_allneu_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_RSM_mean_asis_HVA', 'list_corr_sesspair_asis_HVA', 'list_rate_RRneuron_dr_HVA', 'list_RSM_mean_RRneuron_HVA', 'list_corr_sesspair_HVA'],
                     'list_RSM_mean_asis_HVA': list_RSM_mean_asis_HVA, 'list_corr_sesspair_asis_HVA': list_corr_sesspair_asis_HVA,
                     'list_rate_RRneuron_dr_HVA': list_rate_RRneuron_dr_HVA, 'list_RSM_mean_RRneuron_HVA': list_RSM_mean_RRneuron_HVA, 'list_corr_sesspair_HVA': list_corr_sesspair_HVA}, f)

    print("Ended Process", c_proc.name)

# %%
# RSA within sessions (ABO, RRneuron)

def RSA_withinsess_ABO(slope_ind, target_slope, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # 각 trial type에서 trial sampling할 횟수
    num_trials = 50
    num_trial_types = 119
    num_sess = 32
    n_neu_sampling = 10

    print(f'target slope {target_slope:.1f}')

    # 모든 session에 대해 iteration
    np.random.seed(0) # as-is와 RRneuron, 모든 slope에 대해 뉴런 분할을 동일하게 하기 위함.
    # random.seed(0)

    list_corr_withinsess_asis2 = np.full((num_sess, n_neu_sampling, 3), np.nan) # correlation 측정법 3가지
    list_corr_withinsess2 = np.full((num_sess, n_neu_sampling, 3), np.nan) # correlation 측정법 3가지

    for sess_ind in range(num_sess):

        print(f'sess_ind: {sess_ind}')

        rate = list_rate_all[sess_ind].copy()
        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # delta t 곱해서 spike count로 만들기
        rate = rate * 0.25

        # stm type별 counting dictionary 제작
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # 3D matrix로 만들기
        min_num_trials = np.min(all_stm_counts) # session 4는 trial repeat 수가 균질하지 않으므로 최솟값만 취함 (참고: 47개)

        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(np.arange(-1, 118, 1).astype(int)):
            list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        rate = np.stack(list_rate_tt, axis=2)
        rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x num_trials
    
        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)
        
        list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

        # # trial shuffling
        # rate_shuf = np.zeros_like(rate_sorted)
        # for neu_ind in range(rate_sorted.shape[0]):
        #     shuf_inds = np.random.permutation(rate_sorted.shape[2])
        #     rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() # transpose 주의!
        # rate_sorted = rate_shuf.copy()

        # trial 순서 re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

        # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        # RRneuron var 계산
        var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
        offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

        var_rs_noisy = \
            pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
        var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

        # rate residual RR 계산 & mean과 다시 합하기            
        rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
            * np.sqrt(var_rs_noisy)
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # NaN을 다시 0으로 바꾸기!      

        # # trial 순서 re-randomization
        # for trial_type_ind in range(num_trial_types):
        #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)] 

        # 서로 다른 neuron division에 대해 iteration
        for neu_sample_ind in range(n_neu_sampling):
            print(f'neu_sample_ind = {neu_sample_ind}')
            
            # neuron 분할
            neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
            neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 분할 (참고: int는 버림이다)
            neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
            if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # 이 세션의 뉴런 개수가 홀수일 경우
                neu_div_inds2 = neu_div_inds2[:-1].copy()

            # as-is

            if slope_ind == 0:

                # RSM 반복 제작
                
                if similarity_type == 'geodesic' or similarity_type == 'euclidean':
                    rate_sorted_2d = np.reshape(rate_sorted, (-1, num_trial_types*num_trials))

                    # isomap
                    n_components = 1 # 목표 차원 수
                    # n_components = rate_sorted.shape[0] # 목표 차원 수
                    n_neighbors = 10 # 이웃 점 개수

                    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
                    
                    rate_sorted_2d_isomap = isomap.fit_transform(rate_sorted_2d.T).T
                    dist_matrix = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

                    if similarity_type == 'euclidean':
                        dist_matrix = cdist(rate_sorted_2d_isomap.T, rate_sorted_2d_isomap.T, 'euclidean')
                
                # n_neurons x n_stimuli 2D matrix sampling
                
                tt_pairs = list(combinations(range(min_num_trials), 2))
                # random.shuffle(tt_pairs) # 10000개만 사용될 것이므로 편향이 없게끔 랜덤화.
                n_sampling = np.min([len(tt_pairs), 10000])
                # n_sampling = len(tt_pairs)

                list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                count = 0
                for sampling_ind in range(n_sampling):
                    
                    if similarity_type == 'cos_sim':

                        rate_sampled_trials1_1 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                        rate_sampled_trials1_2 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                        rate_sampled_trials2_1 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                        rate_sampled_trials2_2 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                        RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!
                        RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                        list_RSM_neu1[sampling_ind] = RSM1.copy()
                        list_RSM_neu2[sampling_ind] = RSM2.copy()

                    else:
                        row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                        col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                        RSM = dist_matrix[row_inds, :][:, col_inds].copy()
                        list_RSM[sampling_ind] = RSM.copy()

                RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # trial sampling에 대해 평균 내기
                RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
                bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
        
            # RRneuron

            # RSM 반복 제작
            
            if similarity_type == 'geodesic' or similarity_type == 'euclidean':
                rate_RRneuron_dr_2d = np.reshape(rate_RRneuron_dr, (-1, num_trial_types*num_trials))

                # isomap
                n_components = 1 # 목표 차원 수
                # n_components = rate_sorted.shape[0] # 목표 차원 수
                n_neighbors = 10 # 이웃 점 개수

                isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
                
                rate_RRneuron_dr_2d_isomap = isomap.fit_transform(rate_RRneuron_dr_2d.T).T
                dist_matrix = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

                if similarity_type == 'euclidean':
                    dist_matrix = cdist(rate_RRneuron_dr_2d_isomap.T, rate_RRneuron_dr_2d_isomap.T, 'euclidean')
            
            # n_neurons x n_stimuli 2D matrix sampling
            
            tt_pairs = list(combinations(range(min_num_trials), 2))
            # random.shuffle(tt_pairs) # 10000개만 사용될 것이므로 편향이 없게끔 랜덤화.
            n_sampling = np.min([len(tt_pairs), 10000])
            # n_sampling = len(tt_pairs)

            list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
            list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

            count = 0
            for sampling_ind in range(n_sampling):
                
                if similarity_type == 'cos_sim':
                    rate_sampled_trials1_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                    rate_sampled_trials1_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                    rate_sampled_trials2_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                    rate_sampled_trials2_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                    RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!
                    RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                    # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) # lower triangle도 대칭으로 채우기
                    list_RSM_neu1[sampling_ind] = RSM1.copy()
                    list_RSM_neu2[sampling_ind] = RSM2.copy()

                else:
                    row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                    col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                    RSM = dist_matrix[row_inds, :][:, col_inds].copy()
                    list_RSM[sampling_ind] = RSM.copy()

                count += 1
                if count % 1000 == 0:
                    print(f'count: {count}')

            RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # trial sampling에 대해 평균 내기
            RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
            list_corr_withinsess2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
            bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
            list_corr_withinsess2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
            list_corr_withinsess2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

    # 파일에 저장
    filename = 'RSM_corr_withinsess_ABO_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis2', 'list_corr_withinsess2'], \
                    'list_corr_withinsess_asis2': list_corr_withinsess_asis2, 'list_corr_withinsess2': list_corr_withinsess2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# RSA within sessions (ABO, RRneuron)

def RSA_withinsess_ABO_HVA(slope_ind, target_slope, similarity_type):
    
    ''' similarity_type is 'cos_sim', 'geodesic', or 'euclidean' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    # n_sampling = 100 # 각 trial type에서 trial sampling할 횟수
    num_trials = 50
    num_trial_types = 119
    num_sess = 32
    n_neu_sampling = 10

    print(f'target slope {target_slope:.1f}')

    np.random.seed(0) # as-is와 RRneuron, 모든 slope에 대해 뉴런 분할을 동일하게 하기 위함.
    # random.seed(0)

    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']

    list_corr_withinsess_asis_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_corr_withinsess_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    for area_ind, area in enumerate(list_HVA_names):
                
        # 모든 session에 대해 iteration

        list_corr_withinsess_asis2 = np.full((num_sess, n_neu_sampling, 3), np.nan) # correlation 측정법 3가지
        list_corr_withinsess2 = np.full((num_sess, n_neu_sampling, 3), np.nan) # correlation 측정법 3가지

        for sess_ind in range(num_sess):

            # print(f'area: {area}, sess_ind: {sess_ind}')

            rate = list_rate_all_HVA[area][sess_ind].copy()
            if np.any(rate) == True: # neuron이 있는 경우

                # rate_sorted = rate.sort_index(axis=1)
                stm = rate.columns.copy()

                # delta t 곱해서 spike count로 만들기
                rate = rate * 0.25

                # stm type별 counting dictionary 제작
                all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
                stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
                
                # 3D matrix로 만들기
                min_num_trials = np.min(all_stm_counts) # session 4는 trial repeat 수가 균질하지 않으므로 최솟값만 취함 (참고: 47개)

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
                    rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() # transpose 주의!
                # rate_sorted = rate_shuf.copy()

                # trial 순서 re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

                # RRneuron var 계산
                var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

                # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
                # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
                offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

                var_rs_noisy = \
                    pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                        / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
                var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

                # rate residual RR 계산 & mean과 다시 합하기            
                rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
                # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                    * np.sqrt(var_rs_noisy)
                # print(rate_resid_RRneuron_dr)
                rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
                rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # NaN을 다시 0으로 바꾸기!      

                # # trial 순서 re-randomization
                # for trial_type_ind in range(num_trial_types):
                #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)] 

                # 서로 다른 neuron division에 대해 iteration
                for neu_sample_ind in range(n_neu_sampling):
                    print(f'area: {area}, sess_ind: {sess_ind}, neu_sample_ind = {neu_sample_ind}')
                    
                    # neuron 분할
                    neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
                    neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 분할 (참고: int는 버림이다)
                    neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
                    if neu_div_inds2.shape[0] > neu_div_inds1.shape[0]: # 이 세션의 뉴런 개수가 홀수일 경우
                        neu_div_inds2 = neu_div_inds2[:-1].copy()

                    # as-is

                    if slope_ind == 0:

                        # RSM 반복 제작
                        
                        if similarity_type == 'geodesic' or similarity_type == 'euclidean':
                            rate_sorted_2d = np.reshape(rate_sorted, (-1, num_trial_types*num_trials))

                            # isomap
                            n_components = 1 # 목표 차원 수
                            # n_components = rate_sorted.shape[0] # 목표 차원 수
                            n_neighbors = 10 # 이웃 점 개수

                            isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
                            
                            rate_sorted_2d_isomap = isomap.fit_transform(rate_sorted_2d.T).T
                            dist_matrix = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

                            if similarity_type == 'euclidean':
                                dist_matrix = cdist(rate_sorted_2d_isomap.T, rate_sorted_2d_isomap.T, 'euclidean')
                        
                        # n_neurons x n_stimuli 2D matrix sampling
                        
                        tt_pairs = list(combinations(range(min_num_trials), 2))
                        # random.shuffle(tt_pairs) # 10000개만 사용될 것이므로 편향이 없게끔 랜덤화.
                        n_sampling = np.min([len(tt_pairs), 10000])
                        # n_sampling = len(tt_pairs)

                        list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                        list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                        count = 0
                        for sampling_ind in range(n_sampling):
                            
                            if similarity_type == 'cos_sim':

                                rate_sampled_trials1_1 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                                rate_sampled_trials1_2 = np.squeeze(rate_sorted[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                                rate_sampled_trials2_1 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                                rate_sampled_trials2_2 = np.squeeze(rate_sorted[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                                RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!
                                RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                                list_RSM_neu1[sampling_ind] = RSM1.copy()
                                list_RSM_neu2[sampling_ind] = RSM2.copy()

                            else:
                                row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                                col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                                RSM = dist_matrix[row_inds, :][:, col_inds].copy()
                                list_RSM[sampling_ind] = RSM.copy()

                        RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # trial sampling에 대해 평균 내기
                        RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                        list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
                        bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                        list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                        list_corr_withinsess_asis2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())
                
                    # RRneuron

                    # RSM 반복 제작
                    
                    if similarity_type == 'geodesic' or similarity_type == 'euclidean':
                        rate_RRneuron_dr_2d = np.reshape(rate_RRneuron_dr, (-1, num_trial_types*num_trials))

                        # isomap
                        n_components = 1 # 목표 차원 수
                        # n_components = rate_sorted.shape[0] # 목표 차원 수
                        n_neighbors = 10 # 이웃 점 개수

                        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
                        
                        rate_RRneuron_dr_2d_isomap = isomap.fit_transform(rate_RRneuron_dr_2d.T).T
                        dist_matrix = isomap.dist_matrix_.copy() # 2D geodesic distance matrix

                        if similarity_type == 'euclidean':
                            dist_matrix = cdist(rate_RRneuron_dr_2d_isomap.T, rate_RRneuron_dr_2d_isomap.T, 'euclidean')
                    
                    # n_neurons x n_stimuli 2D matrix sampling
                    
                    tt_pairs = list(combinations(range(min_num_trials), 2))
                    # random.shuffle(tt_pairs) # 10000개만 사용될 것이므로 편향이 없게끔 랜덤화.
                    n_sampling = np.min([len(tt_pairs), 10000])
                    # n_sampling = len(tt_pairs)

                    list_RSM_neu1 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)
                    list_RSM_neu2 = np.zeros((n_sampling, num_trial_types, num_trial_types), dtype=np.float32)

                    count = 0
                    for sampling_ind in range(n_sampling):
                        
                        if similarity_type == 'cos_sim':
                            rate_sampled_trials1_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                            rate_sampled_trials1_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds1, :, tt_pairs[sampling_ind][1]]).copy()
                            rate_sampled_trials2_1 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][0]]).copy() # np.squeeze도 deepcopy 필수!
                            rate_sampled_trials2_2 = np.squeeze(rate_RRneuron_dr[neu_div_inds2, :, tt_pairs[sampling_ind][1]]).copy()

                            RSM1 = np.array(normc(rate_sampled_trials1_1).T) @ np.array(normc(rate_sampled_trials1_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!
                            RSM2 = np.array(normc(rate_sampled_trials2_1).T) @ np.array(normc(rate_sampled_trials2_2)) # 각 trial vector를 단위벡터로 만들고 내적하면 cosine similarity!

                            # RSM_cos = RSM_cos + RSM_cos.T - np.diag(np.diag(RSM_cos)) # lower triangle도 대칭으로 채우기
                            list_RSM_neu1[sampling_ind] = RSM1.copy()
                            list_RSM_neu2[sampling_ind] = RSM2.copy()

                        else:
                            row_inds = np.linspace(tt_pairs[sampling_ind][0], tt_pairs[sampling_ind][0]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)
                            col_inds = np.linspace(tt_pairs[sampling_ind][1], tt_pairs[sampling_ind][1]+(num_trial_types-1)*min_num_trials, num_trial_types, endpoint=True).astype(int)

                            RSM = dist_matrix[row_inds, :][:, col_inds].copy()
                            list_RSM[sampling_ind] = RSM.copy()

                        count += 1
                        if count % 1000 == 0:
                            print(f'count: {count}')

                    RSM_mean_neu1 = np.nanmean(list_RSM_neu1, axis=0) # trial sampling에 대해 평균 내기
                    RSM_mean_neu2 = np.nanmean(list_RSM_neu2, axis=0)
                    list_corr_withinsess2[sess_ind, neu_sample_ind, 0] = spearmanr(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten(), nan_policy='omit').statistic # 모두 NaN인 경우 nanmean 후에도 남아있을 수 있으므로.
                    bool_notnan = np.logical_and(~np.isnan(RSM_mean_neu1.flatten()), ~np.isnan(RSM_mean_neu2.flatten()))
                    list_corr_withinsess2[sess_ind, neu_sample_ind, 1] = np.corrcoef(RSM_mean_neu1.flatten()[bool_notnan], RSM_mean_neu2.flatten()[bool_notnan])[0, 1]
                    list_corr_withinsess2[sess_ind, neu_sample_ind, 2] = cos_sim(RSM_mean_neu1.flatten(), RSM_mean_neu2.flatten())

        list_corr_withinsess_asis_HVA[area] = list_corr_withinsess_asis2.copy()
        list_corr_withinsess_HVA[area] = list_corr_withinsess2.copy()

    # 파일에 저장
    filename = 'RSM_corr_withinsess_ABO_HVA_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_corr_withinsess_asis_HVA', 'list_corr_withinsess_HVA'], \
                    'list_corr_withinsess_asis_HVA': list_corr_withinsess_asis_HVA, 'list_corr_withinsess_HVA': list_corr_withinsess_HVA}, f)
        
    print("Ended Process", c_proc.name)

# %%
# decoding (ABO)
def decode_ABO(sess_ind, decoder_type):

    ''' decoder_type is SVM, logit, Bayesian, RF, kNN '''

    # warning 출력 막기
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

        all_stimuli = np.arange(-1, 118, 1).astype(int) # grayscreen 포함
        probe_stimuli = np.array([5, 12, 24, 34, 36, 44, 47, 78, 83, 87, 104, 111, 114, 115])
        # train_stimuli = all_stimuli[~np.isin(all_stimuli, probe_stimuli)].copy()
        train_stimuli = all_stimuli.copy()

        rate = list_rate_all[sess_ind].copy()
        
        # print(f'sess_ind: {sess_ind}')

        # rate_sorted = rate.sort_index(axis=1)
        stm = rate.columns.copy()

        # delta t 곱해서 spike count로 만들기
        rate = rate * 0.25

        # stm type별 counting dictionary 제작
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
        
        # 3D matrix로 만들기
        min_num_trials = np.min(all_stm_counts) # session 4는 trial repeat 수가 균질하지 않으므로 최솟값만 취함 (참고: 47개)

        list_rate_tt = [None] * num_trial_types
        for trial_type_ind, trial_type in enumerate(all_stimuli):
            list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

        rate = np.stack(list_rate_tt, axis=2)
        rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x min_num_trials

        # # trial shuffling
        # rate_shuf = np.zeros_like(rate_sorted)
        # for neu_ind in range(rate_sorted.shape[0]):
        #     shuf_inds = np.random.permutation(rate_sorted.shape[2])
        #     rate_shuf[neu_ind] = rate_sorted[neu_ind, :, shuf_inds].T.copy() # transpose 주의!
        # rate_sorted = rate_shuf.copy()

        # trial type별 mean & variance 계산
        rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
        rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
            np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)

        list_slopes_dr = list_slopes_all_an_loglog[sess_ind].copy()

        # trial 순서 re-randomization
        for trial_type_ind in range(num_trial_types):
            rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]
        
        # decoding cross-validation (as-is)
        kfold = KFold(n_splits=n_splits)
        stkfold = StratifiedKFold(n_splits=n_splits)

        # 각 trial type 추출
        label_train = np.repeat(all_stimuli, min_num_trials)
        rate_train = pd.DataFrame(rate_sorted.reshape(rate_sorted.shape[0], -1), columns=label_train)
        
        # decoding cross-validation (as-is)
        kfold = KFold(n_splits=n_splits)
        stkfold = StratifiedKFold(n_splits=n_splits)

        list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
        list_accuracy = np.full(n_splits, np.nan)

        if decoder_type in ['SVM', 'logit', 'RF', 'kNN']:

            for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
                X_train, X_test = rate_train.T.iloc[train_index].copy(), rate_train.T.iloc[test_index].copy() # train, test data/label 선언
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
                    clf = KNeighborsClassifier(n_neighbors=30) # 저차원 데이터의 rule of thumb은 root of sample number, 고차원 데이터는 더 작은 값이 좋으므로 50 이내의 값 추천 받음. (GPT) 
                                
                clf.fit(X_train, y_train) # SVC fitting to train data
                
                y_test_pred = clf.predict(X_test) # test data에 대한 predicted label

                # normalized test confusion matrix/test accuracy를 list에 추가
                test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
                test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                list_confusion_test[split_ind] = test_confusion_matrix.copy()

                accuracy = accuracy_score(y_test, y_test_pred)
                list_accuracy[split_ind] = accuracy
                # print(accuracy)
            
                # print(f'split_ind {split_ind}')

        # cross-validation average test confusion matrix/test accuracy 계산
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

            print(f'sess_ind: {sess_ind}, target slope {target_slope:.1f}')
                                
            # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
            rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
            rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

            # RRneuron var 계산
            var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

            # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
            offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

            var_rs_noisy = \
                pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                    / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
            var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

            # rate residual RR 계산 & mean과 다시 합하기
            rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                * np.sqrt(var_rs_noisy)
            # print(rate_resid_RRneuron_dr)
            rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # NaN을 다시 0으로 바꾸기!        

            # # trial 순서 re-randomization
            # for trial_type_ind in range(num_trial_types):
            #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]
            
            # decoding cross-validation (RRneuron)  

            # 각 trial type 추출
            label_train_RRneuron = np.repeat(all_stimuli, min_num_trials)
            rate_train_RRneuron = pd.DataFrame(rate_RRneuron_dr.reshape(rate_RRneuron_dr.shape[0], -1), columns=label_train_RRneuron)
            
            # decoding cross-validation (as-is)
            kfold = KFold(n_splits=n_splits)
            stkfold = StratifiedKFold(n_splits=n_splits)

            list_confusion_test = np.full((n_splits, len(train_stimuli), len(train_stimuli)), np.nan)
            list_accuracy = np.full(n_splits, np.nan)

            if decoder_type in ['SVM', 'logit', 'RF', 'kNN']:

                for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
                    X_train, X_test = rate_train_RRneuron.T.iloc[train_index].copy(), rate_train_RRneuron.T.iloc[test_index].copy() # train, test data/label 선언
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
                        clf = KNeighborsClassifier(n_neighbors=30) # 저차원 데이터의 rule of thumb은 root of sample number, 고차원 데이터는 더 작은 값이 좋으므로 50 이내의 값 추천 받음. (GPT)
                                        
                    clf.fit(X_train, y_train) # SVC fitting to train data
                    
                    y_test_pred = clf.predict(X_test) # test data에 대한 predicted label

                    # normalized test confusion matrix/test accuracy를 list에 추가
                    test_confusion_matrix = confusion_matrix(y_test, y_test_pred)
                    test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                    list_confusion_test[split_ind] = test_confusion_matrix.copy()

                    accuracy = accuracy_score(y_test, y_test_pred)
                    list_accuracy[split_ind] = accuracy
                    # print(accuracy)
                                                
                    # print(f'split_ind {split_ind}')

            # cross-validation average test confusion matrix/test accuracy 계산
            mean_confusion_test = sum(list_confusion_test) / n_splits
            mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
            # print(mean_confusion_test_Bayes.round(3))
            list_mean_confusion_test_RRneuron[slope_ind] = mean_confusion_test.copy()
            
            mean_accuracy = np.mean(list_accuracy)
            # print(round(mean_accuracy, ndigits=3))
            list_mean_accuracy_RRneuron[slope_ind] = mean_accuracy

            print(f'sess_ind: {sess_ind}, target slope {target_slope:.1f}, duration {(time()-start_time)/60:.2f} min')

    # 파일에 저장
    filename = decoder_type + '_decoding_ABO_allstim_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['mean_confusion_test_asis', 'mean_accuracy_asis', 'list_mean_confusion_test_RRneuron', 'list_mean_accuracy_RRneuron'],
                     'mean_confusion_test_asis': mean_confusion_test_asis, 'mean_accuracy_asis': mean_accuracy_asis,
                     'list_mean_confusion_test_RRneuron': list_mean_confusion_test_RRneuron, 'list_mean_accuracy_RRneuron': list_mean_accuracy_RRneuron}, f)
                
    print("Ended Process", c_proc.name)

# %%
# decoding (ABO, HVA)
def decode_ABO_HVA(slope_ind, target_slope, decoder_type):

    ''' decoder_type is SVM or Bayesian '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_sess = 32
    num_trial_types = 119
    num_trials = 50
    n_splits = 10
    t_win = 0.25

    np.random.seed(0) # as-is와 RRneuron의 뉴런 분할 & 모든 slope의 뉴런 분할을 동일하게 하기 위함.

    all_stimuli = np.arange(-1, 118, 1).astype(int) # grayscreen 포함
    probe_stimuli = np.array([5, 12, 24, 34, 36, 44, 47, 78, 83, 87, 104, 111, 114, 115])
    # train_stimuli = all_stimuli[~np.isin(all_stimuli, probe_stimuli)].copy()
    train_stimuli = all_stimuli.copy()

    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']

    list_mean_confusion_test_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_mean_accuracy_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}

    list_mean_confusion_test_RRneuron_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_mean_accuracy_RRneuron_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}

    for area_ind, area in enumerate(list_HVA_names):

        # 모든 session에 대해 iteration

        list_mean_confusion_test = []
        list_mean_accuracy = []

        list_mean_confusion_test_RRneuron = []
        list_mean_accuracy_RRneuron = []

        for sess_ind in range(num_sess):
            # print(f'area {area}, sess_ind: {sess_ind}')

            rate = list_rate_all_HVA[area][sess_ind].copy()

            if np.any(rate) == True: # neuron이 있을 때
                stm = rate.columns.copy()

                # delta t 곱해서 spike count로 만들기
                rate = rate * 0.25

                # stm type별 counting dictionary 제작
                all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
                stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))
                
                # 3D matrix로 만들기
                min_num_trials = np.min(all_stm_counts) # session 4는 trial repeat 수가 균질하지 않으므로 최솟값만 취함 (참고: 47개)

                list_rate_tt = [None] * num_trial_types
                for trial_type_ind, trial_type in enumerate(all_stimuli):
                    list_rate_tt[trial_type_ind] = rate.loc[:, trial_type].iloc[:, :min_num_trials].copy()

                rate = np.stack(list_rate_tt, axis=2)
                rate_sorted = np.transpose(rate, (0, 2, 1)) # num_neurons x num_trial_types x min_num_trials

                # trial type별 mean & variance 계산
                rate_sorted_mean_coll, rate_sorted_var_coll = np.mean(rate_sorted, axis=2), np.var(rate_sorted, axis=2, ddof=1)
                rate_sorted_mean, rate_sorted_var = np.repeat(rate_sorted_mean_coll[:, :, np.newaxis], min_num_trials, axis=2), \
                    np.repeat(rate_sorted_var_coll[:, :, np.newaxis], min_num_trials, axis=2)

                list_slopes_dr = list_slopes_all_an_loglog_HVA[area][sess_ind].copy()

                # trial 순서 re-randomization
                for trial_type_ind in range(num_trial_types):
                    rate_sorted[:, trial_type_ind, :] = rate_sorted[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]
                
                # decoding cross-validation (as-is)
                kfold = KFold(n_splits=n_splits)
                stkfold = StratifiedKFold(n_splits=n_splits)
                
                # slope 하나에서만 as-is도 계산
                if slope_ind == 0:

                    # 각 trial type 추출
                    label_train = np.repeat(all_stimuli, min_num_trials)
                    rate_train = pd.DataFrame(rate_sorted.reshape(rate_sorted.shape[0], -1), columns=label_train)
                    
                    # decoding cross-validation (as-is)
                    kfold = KFold(n_splits=n_splits)
                    stkfold = StratifiedKFold(n_splits=n_splits)

                    list_confusion_test = []
                    list_accuracy = []

                    if decoder_type == 'SVM':
                        for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train.T, label_train)):
                            X_train, X_test = rate_train.T.iloc[train_index].copy(), rate_train.T.iloc[test_index].copy() # train, test data/label 선언
                            y_train, y_test = label_train[train_index].copy(), label_train[test_index].copy()

                            mean_ = X_train.mean(axis=0)
                            X_train = X_train.sub(mean_, axis=1) # train data mean centering
                            X_test = X_test.sub(mean_, axis=1)

                            clf_SVC = svm.SVC(kernel='linear')
                            # clf_SVC = svm.SVC(kernel='poly', degree=3)
                            # clf_SVC = svm.SVC(kernel='rbf')
                            
                            clf_SVC.fit(X_train, y_train) # SVC fitting to train data
                            
                            y_test_pred_SVC = clf_SVC.predict(X_test) # test data에 대한 predicted label

                            # normalized test confusion matrix/test accuracy를 list에 추가
                            test_confusion_matrix = confusion_matrix(y_test, y_test_pred_SVC)
                            test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                            list_confusion_test.append(test_confusion_matrix)

                            accuracy = accuracy_score(y_test, y_test_pred_SVC)
                            list_accuracy.append(accuracy)
                            # print(accuracy)
                        
                            # print(f'split_ind {split_ind}')

                    # cross-validation average test confusion matrix/test accuracy 계산
                    mean_confusion_test = sum(list_confusion_test) / n_splits
                    mean_confusion_test = pd.DataFrame(mean_confusion_test, columns=train_stimuli, index=train_stimuli).fillna(0)
                    # print(mean_confusion_test.round(3))
                    list_mean_confusion_test.append(mean_confusion_test)
                    
                    mean_accuracy = np.mean(list_accuracy)
                    # print(round(mean_accuracy, ndigits=3))
                    list_mean_accuracy.append(mean_accuracy)

                # RRneuron

                print(f'target slope {target_slope:.1f}, area {area}, sess_ind: {sess_ind}')
                                    
                # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

                # RRneuron var 계산
                var_estim_dr = np.nanmean(rate_sorted_var_coll, axis=0)

                # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
                # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
                offset = pow(10, (list_slopes_dr[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

                var_rs_noisy = \
                    pow(10, (np.log10(rate_sorted_var_coll) - list_slopes_dr[1, :])\
                        / list_slopes_dr[0, :] * target_slope + np.log10(np.array(offset))) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
                var_rs_noisy = np.repeat(np.squeeze(var_rs_noisy)[:, :, np.newaxis], min_num_trials, axis=2)

                # rate residual RR 계산 & mean과 다시 합하기
                rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
                # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                rate_resid_RRneuron_dr = rate_sorted_resid_dr / np.sqrt(rate_sorted_var) \
                    * np.sqrt(var_rs_noisy)
                # print(rate_resid_RRneuron_dr)
                rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
                rate_RRneuron_dr[np.isnan(rate_RRneuron_dr)] = 0 # NaN을 다시 0으로 바꾸기!        

                # # trial 순서 re-randomization
                # for trial_type_ind in range(num_trial_types):
                #     rate_RRneuron_dr[:, trial_type_ind, :] = rate_RRneuron_dr[:, trial_type_ind, np.random.choice(range(min_num_trials), min_num_trials, replace=False)]

                # decoding cross-validation (RRneuron)  

                # 각 trial type 추출
                label_train_RRneuron = np.repeat(all_stimuli, min_num_trials)
                rate_train_RRneuron = pd.DataFrame(rate_RRneuron_dr.reshape(rate_RRneuron_dr.shape[0], -1), columns=label_train_RRneuron)
                
                # decoding cross-validation (as-is)
                kfold = KFold(n_splits=n_splits)
                stkfold = StratifiedKFold(n_splits=n_splits)

                list_confusion_test = []
                list_accuracy = []

                if decoder_type == 'SVM':
                    for split_ind, (train_index, test_index) in enumerate(stkfold.split(rate_train_RRneuron.T, label_train_RRneuron)):
                        X_train, X_test = rate_train_RRneuron.T.iloc[train_index].copy(), rate_train_RRneuron.T.iloc[test_index].copy() # train, test data/label 선언
                        y_train, y_test = label_train_RRneuron[train_index].copy(), label_train_RRneuron[test_index].copy()

                        mean_ = X_train.mean(axis=0)
                        X_train = X_train.sub(mean_, axis=1) # train data mean centering
                        X_test = X_test.sub(mean_, axis=1)

                        clf_SVC = svm.SVC(kernel='linear')
                        # clf_SVC = svm.SVC(kernel='poly', degree=3)
                        # clf_SVC = svm.SVC(kernel='rbf')
                        
                        clf_SVC.fit(X_train, y_train) # SVC fitting to train data
                        
                        y_test_pred_SVC = clf_SVC.predict(X_test) # test data에 대한 predicted label

                        # normalized test confusion matrix/test accuracy를 list에 추가
                        test_confusion_matrix = confusion_matrix(y_test, y_test_pred_SVC)
                        test_confusion_matrix = test_confusion_matrix / np.sum(test_confusion_matrix, axis=1, keepdims=True)
                        list_confusion_test.append(test_confusion_matrix)

                        accuracy = accuracy_score(y_test, y_test_pred_SVC)
                        list_accuracy.append(accuracy)
                        # print(accuracy)
                    
                        # print(f'split_ind {split_ind}')

                # cross-validation average test confusion matrix/test accuracy 계산
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

    # 파일에 저장
    filename = decoder_type + '_decoding_ABO_HVA_' + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_mean_confusion_test_HVA', 'list_mean_accuracy_HVA', 'list_mean_confusion_test_RRneuron_HVA', 'list_mean_accuracy_RRneuron_HVA'],
                     'list_mean_confusion_test_HVA': list_mean_confusion_test_HVA, 'list_mean_accuracy_HVA': list_mean_accuracy_HVA,
                     'list_mean_confusion_test_RRneuron_HVA': list_mean_confusion_test_RRneuron_HVA, 'list_mean_accuracy_RRneuron_HVA': list_mean_accuracy_RRneuron_HVA}, f)

    print("Ended Process", c_proc.name)

# %%
# matlab 분석에 필요한 변수들을 .mat 파일로 저장
def save_ABO_variables(sess_ind):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    sess_id = brain_observatory_sessid[sess_ind]
    print(f'sess_ind = {sess_ind}\n{sess_id}')

    sess_name = 'session_' + str(sess_id)
    file_path = 'D:\\Users\\USER\\MATLAB\\Allen_Brain_Neuropixels\\ecephys_cache_dir\\' + sess_name + '\\' + sess_name + '.nwb'
    
    with NWBHDF5IO(file_path, "r", load_namespaces=True) as io:
        nwb = io.read()

        # 각 unit의 brain area 저장
        nwbread_unit_ids = nwb.units.id.data[:].copy()
        # print(f'total unit number {nwbread_unit_ids.shape[0]}')
        
        units_df = nwb.units[:].copy() # 각 unit이 어떤 peak channel에 속하는지 포함
        electrodes_df = nwb.electrodes[:].copy() # 각 channel이 어떤 brain area인지 포함

        # Suppose "peak_channel_id" is how units reference an electrode
        # Make sure it's integer or a matching type so we can join
        units_df["peak_channel_id"] = units_df["peak_channel_id"].astype(int).copy()

        # Rename the index in electrodes_df for easy merging
        electrodes_df = electrodes_df.rename_axis("channel_id").reset_index()

        # Merge on the channel_id vs. the peak_channel_id
        merged_df = pd.merge(
            left=units_df.reset_index(),  # reset_index() so that the "id" is a column
            right=electrodes_df,
            how="left",
            left_on="peak_channel_id",
            right_on="channel_id"
        )

        list_unit_areas = merged_df.loc[:, 'location'].copy()

        # list_unit_areas = []
        # for unit_ind, unit_id in enumerate(nwbread_unit_ids):
        #     print(unit_ind)
        #     # print(nwb.electrodes[:].loc[nwb.units[:].loc[unit_id, 'peak_channel_id'], 'location'])
        #     list_unit_areas.append(nwb.electrodes[:].loc[nwb.units[:].loc[unit_id, 'peak_channel_id'], 'location'])
        # # print(list_unit_areas)

        # .mat 파일로 변수들 저장
        savefile_name = 'nwb_' + str(sess_id) + '.mat'
        savemat(savefile_name, dict(unit_ids=nwb.units.id.data[:],
                                    unit_times_data=nwb.units.spike_times.data[:],
                                    unit_times_idx=nwb.units.spike_times_index.data[:],
                                    unit_loc=np.array(list_unit_areas, dtype=object))) # 마지막 변수는 matlab cell로 저장됨

    # io.close()

    print("Ended Process", c_proc.name)

# %%
# 변수 loading

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

# # RSA across session pairs
# if __name__ == '__main__':

#     with mp.Pool() as pool:
#         list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
#         pool.starmap(RSA_across_sesspairs_ABO, list_inputs)

# # RSA within sessions
# if __name__ == '__main__':

#     with mp.Pool() as pool:
#         list_inputs = [[slope_ind, target_slope, 'cos_sim'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
#         pool.starmap(RSA_withinsess_ABO_HVA, list_inputs)

# # decoding
# if __name__ == '__main__':

#     with mp.Pool(processes=24) as pool:
#         list_inputs = [[sess_ind, 'SVM'] for sess_ind in range(num_sess)]
        
#         pool.starmap(decode_ABO, list_inputs)

# # ABO 변수 저장
# num_sess = 32
# if __name__ == '__main__':

#     with mp.Pool() as pool:
#         list_inputs = [[sess_ind] for sess_ind in np.arange(28, 32).astype(int)]
        
#         pool.starmap(save_ABO_variables, list_inputs)
