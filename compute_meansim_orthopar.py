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
# cosine similarity 계산 함수
def cos_sim(x, y):
    # x, y 각각 1d vector

    # dot_xy = np.dot(x, y)
    # norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)

    # cos_sim = np.dot(normr(np.squeeze(np.squeeze(x).reshape(1, -1))), \
    #                  normr(np.squeeze(np.squeeze(y).reshape(1, -1))))

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# %%
# orthogonal & parallel distance 계산 함수
def compute_orth_par_dist(manifold_name1, manifold_name2, rate_12, rate_sorted_mean_coll):
    mean_vector = rate_sorted_mean_coll.loc[:, manifold_name2] - rate_sorted_mean_coll.loc[:, manifold_name1] # mean vector
    mat1 = rate_12.loc[:, manifold_name1].sub(rate_sorted_mean_coll.loc[:, manifold_name1], axis=0) # trial vector
    mat_orth1 = np.array(mat1.apply(lambda x : np.dot(x, mean_vector), axis=0).div(np.dot(mean_vector, mean_vector)))[:, np.newaxis].T * np.array(mean_vector)[:, np.newaxis]
    mat_par1 = mat_orth1 - mat1

    return mat_orth1, mat_par1

# %%
# ABO Neuropixels mean similarity vs. orthogonal variance (RRneuron) (all neurons)

def compute_meansim_orthopar_ABO_RRneuron(slope_ind, target_slope, similarity_type):

    ''' similarity_type is 'cos_sim', 'geodesic' or 'isomap' '''

    c_proc = mp.current_process()
    print("Running on Process", c_proc.name, "PID", c_proc.pid)

    print(f'target slope = {target_slope:.1f}')

    num_sess = 32
    num_trial_types = 119

    # 모든 session에 대해 iteration

    list_mean_sim2_ABO_one_tt = np.zeros((num_sess, num_trial_types, num_trial_types-1)) # 분석할 trial type 수 고려한 순열
    list_orthopar2_ABO_one_tt = np.zeros((num_sess, num_trial_types, num_trial_types-1, 2)) # ortho, par 2가지
    list_tot_var2_ABO_one_tt = np.zeros((num_sess, num_trial_types, num_trial_types-1, 2))
    for sess_ind in range(num_sess):
        print(f'sess_ind: {sess_ind}')

        rate = list_rate_all[sess_ind].copy()
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

        list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind],
                                        columns=rate_sorted_mean_coll.columns).copy()

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

        rate_sorted_mean_coll[rate_sorted_mean_coll.isna()] = 0  
        rate_sorted_var_coll[rate_sorted_var_coll.isna()] = 0

        # FF 출력해서 의도대로 됐는지 확인
        rate_mean_RRneuron_coll, rate_var_RRneuron_coll = \
            compute_mean_var_trial_collapse(stm_cnt_dict, rate_RRneuron_dr)
        # FF_RRneuron = rate_var_RRneuron_dr.div(rate_mean_RRneuron_dr)
        # print(FF_RRneuron)
        # print(rate_var_RRneuron_dr)

        if similarity_type == 'geodesic':
            # trial type별 mean point들 concatenate
            rate_plus_mean = pd.concat([rate_RRneuron_dr, rate_mean_RRneuron_coll], axis=1)

            # geodesic distance matrix 계산
            n_components = 1 # 목표 차원 수
            # n_components = rate_RRneuron_dr.shape[0] # 목표 차원 수
            n_neighbors = 5 # 이웃 점 개수

            isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
            
            isomap.fit(rate_plus_mean.T)
            mean_dist_mat_RRneuron = isomap.dist_matrix_[rate_RRneuron_dr.shape[1]:, rate_RRneuron_dr.shape[1]:].copy() # mean point들 간의 geodesic distance matrix
                        
        # 모든 trial type에 대해 iteration (~25 min)
        list_mean_sim_one_tt = np.zeros((num_trial_types, num_trial_types-1))
        list_orthopar_one_tt = np.zeros((num_trial_types, num_trial_types-1, 2))
        list_tot_var_one_tt = np.zeros((num_trial_types, num_trial_types-1, 2))
        for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):
            print(f'trial type ind {trial_type_ind}')

            bool_not_tt = rate_sorted_mean_coll.columns != trial_type

            # mean point 간 similarity 계산
            if similarity_type == 'cos_sim':
                list_mean_sim_one_tt[trial_type_ind] = [cos_sim(rate_sorted_mean_coll.loc[:, trial_type], rate_sorted_mean_coll.loc[:, partner_tt]) \
                                                    for partner_tt in rate_sorted_mean_coll.columns[bool_not_tt]]
            else:
                list_mean_sim_one_tt[trial_type_ind] = mean_dist_mat_RRneuron[trial_type_ind, bool_not_tt].copy()

            # partner trial type에 대한 orthogonal variance 계산
            for partner_ind, partner_tt in enumerate(rate_sorted_mean_coll.columns[bool_not_tt]):
                rate_pair = rate_RRneuron_dr.loc[:, [trial_type, partner_tt]].copy()
                label_cnt_dict_pair = dict(zip(np.unique(rate_pair.columns, return_counts=True)[0], np.unique(rate_pair.columns, return_counts=True)[1]))
                rate_sorted_mean_coll_pair, rate_sorted_var_coll_pair = compute_mean_var_trial_collapse(label_cnt_dict_pair, rate_pair)

                mat_orth, mat_par = compute_orth_par_dist(trial_type, partner_tt, rate_pair, rate_sorted_mean_coll_pair)
                list_orthopar_one_tt[trial_type_ind, partner_ind] = [np.var(np.linalg.norm(mat_orth, axis=0), ddof=1), \
                                                            np.var(np.linalg.norm(mat_par, axis=0), ddof=1)]
            
            # orthogonal variance normalization용 total variance 계산
            list_tot_var_one_tt[trial_type_ind] = rate_sorted_var_coll.loc[:, trial_type].sum() # mean이 아니라 sum 주의

        list_mean_sim2_ABO_one_tt[sess_ind] = list_mean_sim_one_tt.copy()
        list_orthopar2_ABO_one_tt[sess_ind] = list_orthopar_one_tt.copy()
        list_tot_var2_ABO_one_tt[sess_ind] = list_tot_var_one_tt.copy()

    # 파일에 저장
    filename = 'meansim_orthopar_ABO_allneu_' + similarity_type + str(slope_ind) + '.pickle'
    with open(filename, "wb") as f:
        pickle.dump({'tree_variables': ['list_mean_sim2_ABO_one_tt', 'list_orthopar2_ABO_one_tt', 'list_tot_var2_ABO_one_tt'],
                        'list_mean_sim2_ABO_one_tt': list_mean_sim2_ABO_one_tt, 'list_orthopar2_ABO_one_tt': list_orthopar2_ABO_one_tt, 'list_tot_var2_ABO_one_tt': list_tot_var2_ABO_one_tt}, f)

    print("Ended Process", c_proc.name)

# %%
# 변수 loading

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
        list_inputs = [[slope_ind, target_slope, 'geodesic'] for slope_ind, target_slope in enumerate(list_target_slopes)]
        
        pool.starmap(compute_meansim_orthopar_ABO_RRneuron, list_inputs)
