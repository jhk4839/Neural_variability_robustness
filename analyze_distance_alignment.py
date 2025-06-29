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
import sympy as sp
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
# cosine similarity 계산 함수
def cos_sim(x, y):
    # x, y 각각 1d vector

    # dot_xy = np.dot(x, y)
    # norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)

    # cos_sim = dot_xy / (norm_x * norm_y)

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# %%
# eigenvalue 계산 함수

# ISOMAP
def compute_eigenvalues_isomap(rate, isomap):
    n_samples = rate.shape[1]
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples # centering matrix
    kernel_matrix = -0.5 * H @ (isomap.dist_matrix_ ** 2) @ H
    eigenvalues, _ = np.linalg.eigh(kernel_matrix)
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    return eigenvalues_sorted

# PCA
def compute_eigenvalues_pca(rate):
    eigenvalues, _ = np.linalg.eigh(np.cov(rate))
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]
    
    return eigenvalues_sorted

# %%
# ABO PC1 analysis (개별 trial type PC1 vs. local/global manifold) (RRneuron)

def compute_cos_sim_pc1_adj_ABO(slope_ind, target_slope, adjacency_type):
    
    '''adjacency_type is either 'cos_sim' or 'geodesic' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    num_sess = 32
    num_trial_types = 119

    # 모든 session에 대해 iteration

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

        # delta t 곱해서 spike count로 만들기
        rate_sorted = rate_sorted * 0.25

        # stm type별 counting dictionary 제작
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

        # trial type별 mean & variance 계산
        rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
        rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

        list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                    columns=rate_sorted_mean_coll.columns).copy()

        if adjacency_type == 'geodesic':
            
            # trial type별 mean point들 concatenate
            rate_plus_mean = pd.concat([rate_sorted, rate_sorted_mean_coll], axis=1)

            # geodesic distance matrix 계산
            n_components = 1 # 목표 차원 수
            n_neighbors = 5 # 이웃 점 개수

            isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
            
            isomap.fit(rate_plus_mean.T)
            mean_dist_mat_asis = isomap.dist_matrix_[rate_sorted.shape[1]:, rate_sorted.shape[1]:].copy() # mean point들 간의 geodesic distance matrix

        # trial type별로 PC1과 인접한 trial type과의 mean vector의 cosine similarity 계산
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
                pc_tt = pca.components_.copy() # 이 trial type의 PC들

                # 가장 가까운 trial type(s)과의 mean vector (plane) 구한 후 cosine similarity 계산

                bool_not_tt = rate_sorted_mean_coll.columns != trial_type

                if adjacency_type == 'cos_sim':
                    # 1. cosine similarity로 adjacency 판정
                    adj_tt_ind = np.argmax(list_RSM_cos_coll_ABO[sess_ind][trial_type_ind, bool_not_tt]) # 자기 자신은 뺌 # mean vector
                    adj_tt_ind_pair = np.argsort(list_RSM_cos_coll_ABO[sess_ind][trial_type_ind, bool_not_tt])[-2:].copy() # mean plane
                elif adjacency_type == 'geodesic':
                    # 2. mean point geodesic distance로 adjacency 판정
                    adj_tt_ind = np.argmin(mean_dist_mat_asis[trial_type_ind, bool_not_tt]) # 자기 자신은 뺌
                    adj_tt_ind_pair = np.argsort(mean_dist_mat_asis[trial_type_ind, bool_not_tt])[:2].copy()
                else:
                    raise Exception('wrong adjacency_type')
                
                if adj_tt_ind >= trial_type_ind:
                    adj_tt_ind = adj_tt_ind + 1 # tt를 제외하기 전 원래 index로 수정
                for ind, tt_ind in enumerate(adj_tt_ind_pair):
                    if tt_ind >= trial_type_ind:
                        adj_tt_ind_pair[ind] = adj_tt_ind_pair[ind] + 1
                adj_tt = rate_sorted_mean_coll.columns[adj_tt_ind]
                adj_tt_pair = rate_sorted_mean_coll.columns[adj_tt_ind_pair]

                # adjacent trial type PC1과의 aligment 계산
                rate_adjtt = rate_sorted.loc[:, adj_tt].copy()
                rate_adjtt_pca = pca.fit_transform(rate_adjtt.T).T
                pc_adjtt = pca.components_.copy() # 이 trial type의 PC들
                list_cos_sim_pc1_adj[trial_type_ind] = np.abs(cos_sim(pc_tt[0], pc_adjtt[0])) # 절댓값

                mean_vector = rate_sorted_mean_coll.iloc[:, trial_type_ind].copy() # 원점 mean vector
                list_cos_sim_pc1_ori[trial_type_ind] = np.abs(cos_sim(pc_tt[0], mean_vector)) # 절댓값

                # 3. global manifold와의 alignment 계산
                list_not_tt = rate_sorted_mean_coll.columns[bool_not_tt].copy()
                rate_pca = pca.fit_transform(rate_sorted.loc[:, list_not_tt].T).T # 모든 trial 포함한 manifold (이 trial type 제외)
                pc1_global = pca.components_[0].copy() # global manifold의 PC1

                list_cos_sim_pc1_global[trial_type_ind] = np.abs(cos_sim(pc_tt[0], pc1_global))

            list_cos_sim_pc1_adj2[sess_ind] = list_cos_sim_pc1_adj.copy()
            list_cos_sim_pc1_ori2[sess_ind] = list_cos_sim_pc1_ori.copy()
            list_cos_sim_pc1_global2[sess_ind] = list_cos_sim_pc1_global.copy()
        
        print(f'target_slope = {target_slope:.1f}')

        # RRneuron

        # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        # RRneuron var 계산
        var_estim_dr = pd.DataFrame(np.zeros((1, rate_sorted_var_coll.shape[1])), \
                                columns=rate_sorted_var_coll.columns) # RRneuron0 var (collapsed)
        for trial_type in rate_sorted_var_coll.columns:
            var_estim_dr.loc[:, trial_type] = \
                np.nanmean(rate_sorted_var.loc[:, trial_type].values.flatten()) # nanmean
        # var_estim_dr = np.repeat(var_estim_dr, all_stm_counts, axis=1) # RRneuron0가 아니면 필요 없음
        # print(var_estim_dr)

        # offset = var_estim_dr.div(rate_sorted_var_coll.pow(target_slope/list_slopes_dr.iloc[0, :], axis=1).mean(axis=0))\
        # .mul(pow(10, target_slope * list_slopes_dr.iloc[1, :] / list_slopes_dr.iloc[0, :])) # collapsed # 산술평균 유지
        offset = pow(10, (list_slopes_dr.iloc[0, :]-target_slope) * np.nanmean(np.log10(rate_sorted_mean_coll), axis=0) + list_slopes_dr.iloc[1, :]) # 기하평균 유지, dataframe.mean()은 default로 skipna=True

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

        # FF 출력해서 의도대로 됐는지 확인
        rate_mean_RRneuron_coll, rate_var_RRneuron_coll = \
            compute_mean_var_trial_collapse(stm_cnt_dict, rate_RRneuron_dr)
        # FF_RRneuron = rate_var_RRneuron_dr.div(rate_mean_RRneuron_dr)
        # print(FF_RRneuron)
        # print(rate_var_RRneuron_dr)

        # trial type별로 PC1과 인접한 trial type과의 mean vector의 cosine similarity 계산
        list_cos_sim_pc1_adj_RRneuron = np.zeros(rate_mean_RRneuron_coll.shape[1])
        list_cos_sim_pc1_ori_RRneuron = np.zeros(rate_mean_RRneuron_coll.shape[1])
        list_cos_sim_pc1_global_RRneuron = np.zeros(rate_mean_RRneuron_coll.shape[1])
        for trial_type_ind, trial_type in enumerate(rate_mean_RRneuron_coll.columns):
            rate_tt_RRneuron = rate_RRneuron_dr.loc[:, trial_type].copy()
            rate_tt_RRneuron_pca = pca.fit_transform(rate_tt_RRneuron.T).T

            pc_tt_RRneuron = pca.components_.copy() # 이 trial type의 PC들
            eig_tt_RRneuron = pca.explained_variance_.copy() # 이 trial type의 eigenvalue들

            # 가장 가까운 trial type과의 mean vector 구한 후 cosine similarity 계산

            bool_not_tt = rate_mean_RRneuron_coll.columns != trial_type

            if adjacency_type == 'cos_sim':
                # 1. cosine similarity로 adjacency 판정
                adj_tt_ind = np.argmax(list_RSM_cos_coll_ABO[sess_ind][trial_type_ind, bool_not_tt]) # 자기 자신은 뺌
                adj_tt_ind_pair = np.argsort(list_RSM_cos_coll_ABO[sess_ind][trial_type_ind, bool_not_tt])[-2:].copy()
            elif adjacency_type == 'geodesic':
                # 2. mean point geodesic distance로 adjacency 판정
                adj_tt_ind = np.argmin(mean_dist_mat_asis[trial_type_ind, bool_not_tt]) # 자기 자신은 뺌
                adj_tt_ind_pair = np.argsort(mean_dist_mat_asis[trial_type_ind, bool_not_tt])[:2].copy()
            else:
                raise Exception('wrong adjacency_type')
            
            if adj_tt_ind >= trial_type_ind:
                adj_tt_ind = adj_tt_ind + 1 # tt를 제외하기 전 원래 index로 수정
            for ind, tt_ind in enumerate(adj_tt_ind_pair):
                if tt_ind >= trial_type_ind:
                    adj_tt_ind_pair[ind] = adj_tt_ind_pair[ind] + 1       
            adj_tt = rate_sorted_mean_coll.columns[adj_tt_ind]
            adj_tt_pair = rate_sorted_mean_coll.columns[adj_tt_ind_pair]

            # adjacent trial type PC1과의 aligment 계산
            rate_adjtt_RRneuron = rate_RRneuron_dr.loc[:, adj_tt].copy()
            rate_adjtt_RRneuron_pca = pca.fit_transform(rate_adjtt_RRneuron.T).T
            pc_adjtt_RRneuron = pca.components_.copy() # 이 trial type의 PC들
            list_cos_sim_pc1_adj_RRneuron[trial_type_ind] = np.abs(cos_sim(pc_tt_RRneuron[0], pc_adjtt_RRneuron[0])) # 절댓값

            mean_vector_RRneuron = rate_mean_RRneuron_coll.iloc[:, trial_type_ind].copy()
            list_cos_sim_pc1_ori_RRneuron[trial_type_ind] = np.abs(cos_sim(pc_tt_RRneuron[0], mean_vector_RRneuron)) # 절댓값

            # 3. global manifold PC1과의 alignment 계산
            list_not_tt = rate_mean_RRneuron_coll.columns[bool_not_tt].copy()
            rate_RRneuron_pca = pca.fit_transform(rate_RRneuron_dr.loc[:, list_not_tt].T).T
            pc1_global_RRneuron = pca.components_[0].copy() # global manifold의 PC1

            list_cos_sim_pc1_global_RRneuron[trial_type_ind] = np.abs(cos_sim(pc_tt_RRneuron[0], pc1_global_RRneuron))

        list_cos_sim_pc1_adj_RRneuron2[sess_ind] = list_cos_sim_pc1_adj_RRneuron.copy()
        list_cos_sim_pc1_ori_RRneuron2[sess_ind] = list_cos_sim_pc1_ori_RRneuron.copy()
        list_cos_sim_pc1_global_RRneuron2[sess_ind] = list_cos_sim_pc1_global_RRneuron.copy()

    # 파일에 저장
    filename = 'align_pc1_ABO_' + adjacency_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_cos_sim_pc1_adj2', 'list_cos_sim_pc1_ori2', 'list_cos_sim_pc1_global2',
                                        'list_cos_sim_pc1_adj_RRneuron2', 'list_cos_sim_pc1_ori_RRneuron2', 'list_cos_sim_pc1_global_RRneuron2'],
                                        'list_cos_sim_pc1_adj2': list_cos_sim_pc1_adj2, 'list_cos_sim_pc1_ori2': list_cos_sim_pc1_ori2, 'list_cos_sim_pc1_global2': list_cos_sim_pc1_global2,
                                        'list_cos_sim_pc1_adj_RRneuron2': list_cos_sim_pc1_adj_RRneuron2, 'list_cos_sim_pc1_ori_RRneuron2': list_cos_sim_pc1_ori_RRneuron2,
                                        'list_cos_sim_pc1_global_RRneuron2': list_cos_sim_pc1_global_RRneuron2}, f)
        
    print("Ended Process", c_proc.name)

# %%
# 변수 loading

# openscope
with open('SVM_prerequisite_variables.pickle', 'rb') as f:
    SVM_prerequisite_variables = pickle.load(f)
    
    list_rate_w1 = SVM_prerequisite_variables['list_rate_w1'].copy()
    list_stm_w1 = SVM_prerequisite_variables['list_stm_w1'].copy()
    list_neu_loc = SVM_prerequisite_variables['list_neu_loc'].copy()
    list_wfdur = SVM_prerequisite_variables['list_wfdur'].copy()
    list_slopes_an_loglog_12 = SVM_prerequisite_variables['list_slopes_an_loglog_12'].copy() # high repeat trial type 주의

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
        list_inputs = [[slope_ind, target_slope, 'geodesic'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(compute_cos_sim_pc1_adj_ABO, list_inputs)
