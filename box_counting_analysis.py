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
from copy import deepcopy as dc
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
from sklearn.neighbors import NearestNeighbors

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
def compute_percent_filled(point_cloud, point_cloud_cri, min_box_size, max_box_size, num_box_sizes=1):
    
    """
    Calculate the box-counting dimension of a point cloud.

    Args:
    - point_cloud: n_features x n_samples array of points
    - point_cloud_cri: n_features x n_samples array of points; criterion of min, max for each axis

    - min_box_size (float): Minimum box size to consider.
    - max_box_size (float): Maximum box size to consider.
    - num_box_sizes (int): Number of box sizes to evaluate.
    
    """
        
    # 각 뉴런별로 min, max 계산
    point_cloud= np.array(point_cloud).copy() # [:, None] indexing을 위해 필요
    point_cloud_cri= np.array(point_cloud_cri).copy()
    axis_ranges = np.concatenate([np.min(point_cloud_cri, axis=1)[:, None]-1, np.max(point_cloud_cri, axis=1)[:, None]+1], axis=1) # across slopes 분석 가능하게 하기 위해 min-1, max+1
    
    # min, max 범위 안에 있는 점들만 추출
    bool_withinrange = np.all((point_cloud >= axis_ranges[:, 0].reshape(-1, 1)) & (point_cloud <= axis_ranges[:, 1].reshape(-1, 1)), axis=0)
    point_cloud = point_cloud[:, bool_withinrange].copy()

    # Generate box sizes
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num=num_box_sizes, endpoint=True, base=10)
    box_ratios = []

    if point_cloud.shape[1] > 0:
        
        for box_size in box_sizes:
            
            # 각 축별 박스 번호
            grid_indices = np.floor((point_cloud-np.min(point_cloud, axis=1).reshape(-1, 1)) / box_size)
            
            # unique한 박스만 추출
            unique_boxes = np.unique(grid_indices, axis=1)

            # %filled 저장
            box_ratios.append(unique_boxes.shape[1]) # across slope로 분모 통일하기 때문에 분자만 필요

    else:
        print('No points within min, max of point_cloud_cri')
        box_ratios = [0] * num_box_sizes

    log_box_sizes = np.log10(box_sizes)

    return log_box_sizes, box_ratios

# %%
# %filled analysis (ABO, RRneuron)
def compute_filling_boxcount(slope_ind, target_slope):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_sess = 32
    num_box_sizes = 1

    # 모든 session에 대해 iteration
    list_box_ratios_asis = np.zeros((num_sess, num_box_sizes))
    list_box_ratios_RRneuron = np.zeros((num_sess, num_box_sizes))

    list_box_ratios_asis_isomap = np.zeros((num_sess, num_box_sizes))
    list_box_ratios_RRneuron_isomap = np.zeros((num_sess, num_box_sizes))

    for sess_ind, rate in enumerate(list_rate_all):
        # if sess_ind >= 2:
            print(f'target_slope = {target_slope:.1f}, session index: {sess_ind}')
            
            rate_sorted = rate.sort_index(axis=1)
            stm = rate_sorted.columns.copy()

            # delta t 곱해서 spike count로 만들기
            rate_sorted = rate_sorted * 0.25
            # print(rate_sorted.shape[0])

            # stm type별 counting dictionary 제작
            all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
            stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

            # trial type별 mean & variance 계산
            rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
            rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

            list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                        columns=rate_sorted_mean_coll.columns).copy()
            
            box_size = 1

            # box count 저장
            _, box_ratios = compute_percent_filled(rate_sorted, rate_sorted, min_box_size=box_size, max_box_size=box_size)
            list_box_ratios_asis[sess_ind] = box_ratios.copy()

            # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
            rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
            rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
            
            # RRneuron

            # RRneuron var 계산
            var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                    columns=rate_sorted_var_coll.columns) # RRneuron0 var (collapsed)
            for trial_type in rate_sorted_var_coll.columns:
                var_estim_dr.loc[:, trial_type] = \
                    np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean

            # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
            offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.log10(rate_sorted_mean_coll.mean(axis=0)) + list_slopes_dr.iloc[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True
            offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr.iloc[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

            var_rs_noisy = \
                pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
                    .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
            var_rs_noisy = np.repeat(var_rs_noisy, all_stm_counts, axis=1)

            # rate residual RR 계산 & mean과 다시 합하기            
            rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                .mul(np.sqrt(var_rs_noisy))
            # print(rate_resid_RRneuron_dr)
            rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # NaN을 다시 0으로 바꾸기!

            # FF 출력해서 의도대로 됐는지 확인
            rate_mean_RRneuron_coll, _ = compute_mean_var_trial_collapse(stm_cnt_dict, rate_RRneuron_dr)      

            # box count 저장
            box_size = 1
            _, box_ratios = compute_percent_filled(rate_RRneuron_dr, rate_sorted, min_box_size=box_size, max_box_size=box_size)
            list_box_ratios_RRneuron[sess_ind] = box_ratios.copy()

    # 파일에 저장
    filename = 'filling_boxcount_ABO_' + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_box_ratios_asis', 'list_box_ratios_RRneuron', 'list_box_ratios_asis_isomap', 'list_box_ratios_RRneuron_isomap'],
                     'list_box_ratios_asis': list_box_ratios_asis, 'list_box_ratios_RRneuron': list_box_ratios_RRneuron,
                     'list_box_ratios_asis_isomap': list_box_ratios_asis_isomap, 'list_box_ratios_RRneuron_isomap': list_box_ratios_RRneuron_isomap}, f)
        
    print("Ended Process", c_proc.name)

# %%
# overlap consistency (figure 3)
def compute_overlap_stimpairs_consis(sess_ind):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = [0, 1, 2]
    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    num_sampling = 10 # 뉴런 랜덤 분할 횟수

    print(f'sess_ind: {sess_ind}')
    
    rate_sorted = list_rate_all[sess_ind].sort_index(axis=1)
    stm = rate_sorted.columns.copy()
    num_neurons = rate_sorted.shape[0]
    
    np.random.seed(0)

    # delta t 곱해서 spike count로 만들기
    rate_sorted = rate_sorted * 0.25 # 참고: trial type sorting은 이미 돼 있음

    # stm type별 counting dictionary 제작
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # trial type별 mean & variance 계산
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()

    list_overlap_asis2 = np.full((num_sampling, 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
    list_overlap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan)
    list_gap_asis2 = np.full((num_sampling, 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
    list_gap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan)
    for sampling_ind in range(num_sampling):
        # print(f'sess_ind = {sess_ind}, sampling ind = {sampling_ind}')

        # neuron 분할
        neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
        neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 분할 (참고: int는 버림이다)
        neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
        list_neu_div_inds = dc([neu_div_inds1, neu_div_inds2])
        
        # 모든 stimulus pair에서 overlap 계산

        # 각 stimulus마다 내부 pairwise distance로 기준 잡기
        list_pwdist = np.zeros((2, num_trial_types, 2)) # 뉴런 subset 2개, 5%/mean distance 2개
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            pwdist_tt1 = cdist(rate_sorted.loc[neu_div_inds1, trial_type].T, rate_sorted.loc[neu_div_inds1, trial_type].T, 'euclidean')
            pwdist_tt2 = cdist(rate_sorted.loc[neu_div_inds2, trial_type].T, rate_sorted.loc[neu_div_inds2, trial_type].T, 'euclidean')
            list_pwdist[0, trial_type_ind, 0] = np.percentile(pwdist_tt1, 5)
            list_pwdist[0, trial_type_ind, 1] = np.mean(pwdist_tt1)
            list_pwdist[1, trial_type_ind, 0] = np.percentile(pwdist_tt2, 5)
            list_pwdist[1, trial_type_ind, 1] = np.mean(pwdist_tt2)

        list_overlap_asis = np.full((2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
        list_gap_asis = np.full((2, num_trial_types, num_trial_types), np.nan)
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            for div_ind in range(2):
                n_neighbors = 5
                nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                
                rate_tt = rate_sorted.loc[list_neu_div_inds[div_ind], trial_type].copy()
                rate_rest = rate_sorted.loc[list_neu_div_inds[div_ind]].copy()
                nbrs.fit(rate_tt.T)
                nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
                nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

                pwdist_temp = list_pwdist[div_ind, trial_type_ind, 0].copy()
                overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
                list_overlap_asis[div_ind, trial_type_ind] = overlap.copy()

                pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                list_gap_asis[div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]
                        
        list_overlap_asis2[sampling_ind] = list_overlap_asis.copy()
        list_gap_asis2[sampling_ind] = list_gap_asis.copy()

        # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
        
        list_overlap_RRneuron2 = np.full((len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
        list_gap_RRneuron2 = np.full((len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
        for slope_ind, target_slope in enumerate(list_target_slopes):
                        
            # RRneuron
            
            print(f'sampling ind = {sampling_ind}, target slope = {target_slope:.1f}')

            # RRneuron var 계산
            var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                    columns=rate_sorted_var_coll.columns) # RRneuron0 var (collapsed)
            for trial_type in rate_sorted_var_coll.columns:
                var_estim_dr.loc[:, trial_type] = \
                    np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
            # var_estim_dr = np.repeat(var_estim_dr, all_label_counts, axis=1) # RRneuron0가 아니면 필요 없음
            # print(var_estim_dr)

            # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
            # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
            offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.log10(rate_sorted_mean_coll).mean(axis=0) + list_slopes_dr.iloc[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

            var_rs_noisy = \
                pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
                    .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
            var_rs_noisy = np.repeat(var_rs_noisy, all_stm_counts, axis=1)

            # rate residual RR 계산 & mean과 다시 합하기            
            rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
            # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
            rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                .mul(np.sqrt(var_rs_noisy))
            # print(rate_resid_RRneuron_dr)
            rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
            rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # NaN을 다시 0으로 바꾸기!

            # 모든 stimulus pair에서 overlap 계산

            # 각 stimulus마다 내부 pairwise distance로 기준 잡기
            list_pwdist = np.zeros((2, num_trial_types, 2)) # 뉴런 subset 2개, 5%/mean distance 2개
            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                pwdist_tt1 = cdist(rate_RRneuron_dr.loc[neu_div_inds1, trial_type].T, rate_RRneuron_dr.loc[neu_div_inds1, trial_type].T, 'euclidean')
                pwdist_tt2 = cdist(rate_RRneuron_dr.loc[neu_div_inds2, trial_type].T, rate_RRneuron_dr.loc[neu_div_inds2, trial_type].T, 'euclidean')
                list_pwdist[0, trial_type_ind, 0] = np.percentile(pwdist_tt1, 5)
                list_pwdist[0, trial_type_ind, 1] = np.mean(pwdist_tt1)
                list_pwdist[1, trial_type_ind, 0] = np.percentile(pwdist_tt2, 5)
                list_pwdist[1, trial_type_ind, 1] = np.mean(pwdist_tt2)

            for trial_type_ind, trial_type in enumerate(all_stm_unique):
                for div_ind in range(2):
                    n_neighbors = 5
                    nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                    
                    rate_tt = rate_RRneuron_dr.loc[list_neu_div_inds[div_ind], trial_type].copy()
                    rate_rest = rate_RRneuron_dr.loc[list_neu_div_inds[div_ind]].copy()
                    nbrs.fit(rate_tt.T)
                    nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
                    nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

                    pwdist_temp = list_pwdist[div_ind, trial_type_ind, 0].copy()
                    overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
                    list_overlap_RRneuron2[slope_ind, div_ind, trial_type_ind] = overlap.copy()

                    pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                    gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                    list_gap_RRneuron2[slope_ind, div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]

        list_overlap_RRneuron3[sampling_ind] = list_overlap_RRneuron2.copy()
        list_gap_RRneuron3[sampling_ind] = list_gap_RRneuron2.copy()

    # 파일에 저장
    filename = 'overlap_nbr_stimpairs_consis_ABO_' + str(sess_ind) +  '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_overlap_asis2', 'list_overlap_RRneuron3', 'list_gap_asis2', 'list_gap_RRneuron3'],
                     'list_overlap_asis2': list_overlap_asis2, 'list_overlap_RRneuron3': list_overlap_RRneuron3,
                     'list_gap_asis2': list_gap_asis2, 'list_gap_RRneuron3': list_gap_RRneuron3}, f)
        
    print("Ended Process", c_proc.name)

# %%
# overlap consistency (figure 3)
def compute_overlap_stimpairs_consis_HVA(sess_ind):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = [0, 1, 2]
    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119
    num_sampling = 10 # 뉴런 랜덤 분할 횟수

    np.random.seed(0)

    print(f'sess_ind: {sess_ind}')

    list_HVA_names = ['VISl', 'VISrl', 'VISal', 'VISpm', 'VISam']

    list_overlap_asis_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_overlap_RRneuron_HVA2 = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_gap_asis_HVA = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    list_gap_RRneuron_HVA2 = {hva: np.empty(0, dtype=object) for hva in list_HVA_names}
    for area_ind, area in enumerate(list_HVA_names):

        rate = list_rate_all_HVA[area][sess_ind].copy()
        
        if np.any(rate) > 0: # neuron이 있다면.
            rate_sorted = rate.sort_index(axis=1)
            stm = rate_sorted.columns.copy()
            num_neurons = rate_sorted.shape[0]
            
            # delta t 곱해서 spike count로 만들기
            rate_sorted = rate_sorted * 0.25 # 참고: trial type sorting은 이미 돼 있음

            # stm type별 counting dictionary 제작
            all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
            stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

            # trial type별 mean & variance 계산
            rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
            rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

            list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog_HVA[area][sess_ind], \
                                            columns=rate_sorted_mean_coll.columns).copy()

            list_overlap_asis2 = np.full((num_sampling, 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
            list_overlap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan)
            list_gap_asis2 = np.full((num_sampling, 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
            list_gap_RRneuron3 = np.full((num_sampling, len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan)
            for sampling_ind in range(num_sampling):
                # print(f'sess_ind = {sess_ind}, sampling ind = {sampling_ind}')

                # neuron 분할
                neu_inds_permuted = np.random.permutation(range(rate_sorted.shape[0]))
                neu_div_inds1 = neu_inds_permuted[:int(rate_sorted.shape[0]/2)].copy() # 5:5 분할 (참고: int는 버림이다)
                neu_div_inds2 = neu_inds_permuted[int(rate_sorted.shape[0]/2):].copy()
                list_neu_div_inds = dc([neu_div_inds1, neu_div_inds2])
                
                # 모든 stimulus pair에서 overlap 계산

                # 각 stimulus마다 내부 pairwise distance로 기준 잡기
                list_pwdist = np.zeros((2, num_trial_types, 2)) # 뉴런 subset 2개, 5%/mean distance 2개
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    pwdist_tt1 = cdist(rate_sorted.loc[neu_div_inds1, trial_type].T, rate_sorted.loc[neu_div_inds1, trial_type].T, 'euclidean')
                    pwdist_tt2 = cdist(rate_sorted.loc[neu_div_inds2, trial_type].T, rate_sorted.loc[neu_div_inds2, trial_type].T, 'euclidean')
                    list_pwdist[0, trial_type_ind, 0] = np.percentile(pwdist_tt1, 5)
                    list_pwdist[0, trial_type_ind, 1] = np.mean(pwdist_tt1)
                    list_pwdist[1, trial_type_ind, 0] = np.percentile(pwdist_tt2, 5)
                    list_pwdist[1, trial_type_ind, 1] = np.mean(pwdist_tt2)

                list_overlap_asis = np.full((2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
                list_gap_asis = np.full((2, num_trial_types, num_trial_types), np.nan)
                for trial_type_ind, trial_type in enumerate(all_stm_unique):
                    for div_ind in range(2):
                        n_neighbors = 5
                        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                        
                        rate_tt = rate_sorted.loc[list_neu_div_inds[div_ind], trial_type].copy()
                        rate_rest = rate_sorted.loc[list_neu_div_inds[div_ind]].copy()
                        nbrs.fit(rate_tt.T)
                        nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
                        nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

                        pwdist_temp = list_pwdist[div_ind, trial_type_ind, 0].copy()
                        overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
                        list_overlap_asis[div_ind, trial_type_ind] = overlap.copy()

                        # gap = np.min(nbr_dist, axis=1).groupby(level=0).min()
                        # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                        # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                        # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                        # # print(gap.shape)
                        # # list_gap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[div_ind, all_stm_unique != trial_type, 1]
                        # list_gap_asis[div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]
                                
                list_overlap_asis2[sampling_ind] = list_overlap_asis.copy()
                list_gap_asis2[sampling_ind] = list_gap_asis.copy()

                # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
                rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
                rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
                
                list_overlap_RRneuron2 = np.full((len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
                list_gap_RRneuron2 = np.full((len(list_target_slopes), 2, num_trial_types, num_trial_types), np.nan) # 뉴런 subset 2개
                for slope_ind, target_slope in enumerate(list_target_slopes):
                                
                    # RRneuron
                    
                    print(f'{area}, sampling ind = {sampling_ind}, target slope = {target_slope:.1f}')

                    # RRneuron var 계산
                    var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                            columns=rate_sorted_var_coll.columns) # RRneuron0 var (collapsed)
                    for trial_type in rate_sorted_var_coll.columns:
                        var_estim_dr.loc[:, trial_type] = \
                            np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
                    # var_estim_dr = np.repeat(var_estim_dr, all_label_counts, axis=1) # RRneuron0가 아니면 필요 없음
                    # print(var_estim_dr)

                    # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
                    # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
                    offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.log10(rate_sorted_mean_coll).mean(axis=0) + list_slopes_dr.iloc[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

                    var_rs_noisy = \
                        pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
                            .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
                    var_rs_noisy = np.repeat(np.array(var_rs_noisy), all_stm_counts, axis=1) # numpy 버전 이슈로 ndarray로 바꿔줘야 함

                    # rate residual RR 계산 & mean과 다시 합하기            
                    rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
                    # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                    #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
                    rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
                        .mul(np.sqrt(var_rs_noisy))
                    # print(rate_resid_RRneuron_dr)
                    rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
                    rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # NaN을 다시 0으로 바꾸기!

                    # 모든 stimulus pair에서 overlap 계산

                    # 각 stimulus마다 내부 pairwise distance로 기준 잡기
                    list_pwdist = np.zeros((2, num_trial_types, 2)) # 뉴런 subset 2개, 5%/mean distance 2개
                    for trial_type_ind, trial_type in enumerate(all_stm_unique):
                        pwdist_tt1 = cdist(rate_RRneuron_dr.loc[neu_div_inds1, trial_type].T, rate_RRneuron_dr.loc[neu_div_inds1, trial_type].T, 'euclidean')
                        pwdist_tt2 = cdist(rate_RRneuron_dr.loc[neu_div_inds2, trial_type].T, rate_RRneuron_dr.loc[neu_div_inds2, trial_type].T, 'euclidean')
                        list_pwdist[0, trial_type_ind, 0] = np.percentile(pwdist_tt1, 5)
                        list_pwdist[0, trial_type_ind, 1] = np.mean(pwdist_tt1)
                        list_pwdist[1, trial_type_ind, 0] = np.percentile(pwdist_tt2, 5)
                        list_pwdist[1, trial_type_ind, 1] = np.mean(pwdist_tt2)

                    for trial_type_ind, trial_type in enumerate(all_stm_unique):
                        for div_ind in range(2):
                            n_neighbors = 5
                            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
                            
                            rate_tt = rate_RRneuron_dr.loc[list_neu_div_inds[div_ind], trial_type].copy()
                            rate_rest = rate_RRneuron_dr.loc[list_neu_div_inds[div_ind]].copy()
                            nbrs.fit(rate_tt.T)
                            nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
                            nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

                            pwdist_temp = list_pwdist[div_ind, trial_type_ind, 0].copy()
                            overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
                            list_overlap_RRneuron2[slope_ind, div_ind, trial_type_ind] = overlap.copy()

                            # gap = np.min(nbr_dist, axis=1).groupby(level=0).min()
                            # # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=np.repeat(all_stm_unique[all_stm_unique != trial_type], all_stm_counts[all_stm_unique != trial_type]))
                            # pwdist_mat = pd.DataFrame(cdist(rate_tt.T, rate_rest.T, 'euclidean'), columns=rate_rest.columns)
                            # gap = pwdist_mat.T.groupby(pwdist_mat.columns).min().min(axis=1)
                            # # print(gap.shape)
                            # # list_gap_asis[div_ind, trial_type_ind, all_stm_unique != trial_type] = gap/list_pwdist[div_ind, all_stm_unique != trial_type, 1]
                            # list_gap_RRneuron2[slope_ind, div_ind, trial_type_ind] = gap/list_pwdist[div_ind, :, 1]

                list_overlap_RRneuron3[sampling_ind] = list_overlap_RRneuron2.copy()
                list_gap_RRneuron3[sampling_ind] = list_gap_RRneuron2.copy()

            list_overlap_asis_HVA[area] = list_overlap_asis2.copy()
            list_gap_asis_HVA[area] = list_gap_asis2.copy()
            list_overlap_RRneuron_HVA2[area] = list_overlap_RRneuron3.copy()
            list_gap_RRneuron_HVA2[area] = list_gap_RRneuron3.copy()

    # 파일에 저장
    filename = 'overlap_nbr_stimpairs_consis_ABO_HVA_' + str(sess_ind) +  '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_overlap_asis_HVA', 'list_overlap_RRneuron_HVA2', 'list_gap_asis_HVA', 'list_gap_RRneuron_HVA2'],
                     'list_overlap_asis_HVA': list_overlap_asis_HVA, 'list_overlap_RRneuron_HVA2': list_overlap_RRneuron_HVA2,
                     'list_gap_asis_HVA': list_gap_asis_HVA, 'list_gap_RRneuron_HVA2': list_gap_RRneuron_HVA2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# overlap of stimulus pairs (figure 4)
def compute_overlap_stimpairs(sess_ind):

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
    num_trial_types = 119

    print(f'sess_ind: {sess_ind}')
    
    rate_sorted = list_rate_all[sess_ind].sort_index(axis=1)
    stm = rate_sorted.columns.copy()
    num_neurons = rate_sorted.shape[0]

    # delta t 곱해서 spike count로 만들기
    rate_sorted = rate_sorted * 0.25 # 참고: trial type sorting은 이미 돼 있음

    # stm type별 counting dictionary 제작
    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # trial type별 mean & variance 계산
    rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
    rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

    list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()

    # inter-mean geodesic distance로 인접한 trial type 판정

    # trial type별 mean point들 concatenate
    rate_plus_mean = pd.concat([rate_sorted, rate_sorted_mean_coll], axis=1)

    # geodesic distance matrix 계산
    n_components = 1 # 목표 차원 수
    # n_components = rate.shape[0] # 목표 차원 수
    n_neighbors = 5 # 이웃 점 개수

    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    
    isomap.fit(rate_plus_mean.T)
    mean_dist_mat_asis = isomap.dist_matrix_[rate_sorted.shape[1]:, rate_sorted.shape[1]:].copy() # mean point들 간의 geodesic distance matrix

    # 각 stimulus마다 내부 pairwise distance로 기준 잡기
    pwdist_thr = 5
    list_pwdist = np.zeros((num_trial_types, 3))
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        pwdist = cdist(rate_sorted.loc[:, trial_type].T, rate_sorted.loc[:, trial_type].T, 'euclidean')
        list_pwdist[trial_type_ind, 0] = np.percentile(pwdist, pwdist_thr)
        list_pwdist[trial_type_ind, 1] = np.percentile(pwdist, 100-pwdist_thr)
        list_pwdist[trial_type_ind, 2] = np.mean(pwdist)

    list_overlap_asis = np.full((num_trial_types, num_trial_types), np.nan)
    list_gap_asis = np.full((num_trial_types, num_trial_types), np.nan)
    for trial_type_ind, trial_type in enumerate(all_stm_unique):
        n_neighbors = 5
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        
        rate_tt = rate_sorted.loc[:, trial_type].copy()
        rate_rest = rate_sorted.copy()
        nbrs.fit(rate_tt.T)
        nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
        nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

        # 3. % (knn <= threshold)
        pwdist_temp = list_pwdist[trial_type_ind, 0].copy()
        overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
        list_overlap_asis[trial_type_ind] = overlap.copy()

        gap = np.min(nbr_dist, axis=1).groupby(level=0).quantile(0.05)
        list_gap_asis[trial_type_ind] = gap/list_pwdist[:, 2]

    # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
    rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
    rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan
    
    list_overlap_RRneuron2 = np.full((len(list_target_slopes), num_trial_types, num_trial_types), np.nan)
    list_gap_RRneuron2 = np.full((len(list_target_slopes), num_trial_types, num_trial_types), np.nan)
    for slope_ind, target_slope in enumerate(list_target_slopes):
                    
        # RRneuron
        
        print(f'target slope = {target_slope:.1f}')

        # RRneuron var 계산
        var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                columns=rate_sorted_var_coll.columns) # RRneuron0 var (collapsed)
        for trial_type in rate_sorted_var_coll.columns:
            var_estim_dr.loc[:, trial_type] = \
                np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
        # var_estim_dr = np.repeat(var_estim_dr, all_label_counts, axis=1) # RRneuron0가 아니면 필요 없음
        # print(var_estim_dr)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
        offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.log10(rate_sorted_mean_coll).mean(axis=0) + list_slopes_dr.iloc[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

        var_rs_noisy = \
            pow(10, np.log10(rate_sorted_var_coll).sub(list_slopes_dr.iloc[1, :], axis=1)\
                .div(list_slopes_dr.iloc[0, :], axis=1).mul(target_slope).add(np.log10(np.array(offset)), axis=1)) # broadcasting하려면 Series이거나 ndarray여야 함 # collapsed
        var_rs_noisy = np.repeat(var_rs_noisy, all_stm_counts, axis=1)

        # rate residual RR 계산 & mean과 다시 합하기            
        rate_sorted_resid_dr = rate_sorted - rate_sorted_mean
        # rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
        #     .mul(np.sqrt(rate_sorted_mean)).mul(np.sqrt(FF_estim_dr), axis=1)
        rate_resid_RRneuron_dr = rate_sorted_resid_dr.div(np.sqrt(rate_sorted_var))\
            .mul(np.sqrt(var_rs_noisy))
        # print(rate_resid_RRneuron_dr)
        rate_RRneuron_dr = rate_sorted_mean + rate_resid_RRneuron_dr
        rate_RRneuron_dr[rate_RRneuron_dr.isna()] = 0 # NaN을 다시 0으로 바꾸기!

        # 각 stimulus마다 내부 pairwise distance로 기준 잡기
        list_pwdist = np.zeros((num_trial_types, 3))
        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            pwdist = cdist(rate_RRneuron_dr.loc[:, trial_type].T, rate_RRneuron_dr.loc[:, trial_type].T, 'euclidean')
            list_pwdist[trial_type_ind, 0] = np.percentile(pwdist, pwdist_thr)
            list_pwdist[trial_type_ind, 1] = np.percentile(pwdist, 100-pwdist_thr)
            list_pwdist[trial_type_ind, 2] = np.mean(pwdist)

        for trial_type_ind, trial_type in enumerate(all_stm_unique):
            n_neighbors = 5
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            
            rate_tt = rate_RRneuron_dr.loc[:, trial_type].copy()
            rate_rest = rate_RRneuron_dr.copy()
            nbrs.fit(rate_tt.T)
            nbr_dist, nbr_inds = nbrs.kneighbors(rate_rest.T) # n_query x n_neighbors
            nbr_dist = pd.DataFrame(nbr_dist, index=rate_rest.columns)

            pwdist_temp = list_pwdist[trial_type_ind, 0].copy()
            overlap = (np.min(nbr_dist, axis=1) <= pwdist_temp).groupby(level=0).mean()
            list_overlap_RRneuron2[slope_ind, trial_type_ind] = overlap.copy()
                       
            gap = np.min(nbr_dist, axis=1).groupby(level=0).quantile(0.05)
            list_gap_RRneuron2[slope_ind, trial_type_ind] = gap/list_pwdist[:, 2]

    # 파일에 저장
    filename = 'overlap_nbr_stimpairs_ABO_' + str(sess_ind) +  '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['mean_dist_mat_asis', 'list_overlap_asis', 'list_overlap_RRneuron2', 'list_gap_asis', 'list_gap_RRneuron2'],
                     'mean_dist_mat_asis': mean_dist_mat_asis, 'list_overlap_asis': list_overlap_asis, 'list_overlap_RRneuron2': list_overlap_RRneuron2,
                     'list_gap_asis': list_gap_asis, 'list_gap_RRneuron2': list_gap_RRneuron2}, f)
        
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

    sess_inds_qual_all = resp_matrix_ep_RS_all['sess_inds_qual_all'].copy()

# %%
# multiprocessing
list_target_slopes = np.linspace(0, 2, 21, endpoint=True)

# # filling box count
# if __name__ == '__main__':

#     with mp.Pool() as pool:
#         list_inputs = [[slope_ind, target_slope] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
#         pool.starmap(compute_filling_boxcount, list_inputs)

# overlap between stimulus pairs
num_sess = 32
if __name__ == '__main__':

    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_overlap_stimpairs, list_inputs)