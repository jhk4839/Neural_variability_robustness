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
from statannot import add_stat_annotation

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
# cosine similarity 계산 함수
def cos_sim(x, y):
    # x, y 각각 1d vector

    # NaN 제거
    x, y = np.array(x), np.array(y)
    bool_notnan = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[bool_notnan].copy(), y[bool_notnan].copy()

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
# orthogonal & parallel distance 계산 함수
def compute_orth_par_dist(manifold_name1, manifold_name2, rate_12, rate_sorted_mean_coll):
    mean_vector = rate_sorted_mean_coll.loc[:, manifold_name2] - rate_sorted_mean_coll.loc[:, manifold_name1] # mean vector
    mat1 = rate_12.loc[:, manifold_name1].sub(rate_sorted_mean_coll.loc[:, manifold_name1], axis=0) # trial vector
    mat_orth1 = np.array(mat1.apply(lambda x : np.dot(x, mean_vector), axis=0).div(np.dot(mean_vector, mean_vector)))[:, np.newaxis].T * np.array(mean_vector)[:, np.newaxis]
    mat_par1 = mat_orth1 - mat1

    return mat_orth1, mat_par1

# %%
def weighted_angle(A_left, B_left, A_sing, B_sing):
    
    """
    Compute the weighted angle between two subspaces spanned by the columns of A and B. (Sekmen, 2024)
    
    Parameters:
        A (ndarray): left singular vector matrix for the first subspace.
        B (ndarray): left singular vector matrix for the second subspace.
        A_sing (ndarray): singular values for the first subspace.
        B_sing (ndarray): singular values for the second subspace.

    Returns:
        w_ang (scalar): Weighted angle (in radians) between the two subspaces.
    """

    if B_left.shape[0] > 1: # basis가 2개 이상인 경우
        Q = (A_left@A_sing).T @ (B_left@B_sing)
        try:
            _, S_Q, _ = np.linalg.svd(Q)
        except:
            S_Q = np.full(np.min(Q.shape), np.nan)
        # print(f'singular values: {s}')
        # print(s[np.logical_or(s < -1, s > 1)])

        # Clamp singular values to the interval [-1, 1] to avoid numerical issues
        S_Q = np.clip(S_Q, -1.0, 1.0) # 어차피 singular value >= 0으로 나와서 음수는 없긴 함.
        A_sing = np.clip(A_sing, -1.0, 1.0)
        B_sing = np.clip(B_sing, -1.0, 1.0)

        w_ang = np.arccos(np.sum(S_Q) / np.trace(A_sing.T @ B_sing))

    else: # basis가 1개인 경우 (isomap 2d에서 1d tangent line 구할 때)
        S_Q = (A_left@A_sing).T @ (B_left@B_sing) # scalar

        # Clamp singular values to the interval [-1, 1] to avoid numerical issues
        S_Q = np.clip(S_Q, -1.0, 1.0) # 어차피 singular value >= 0으로 나와서 음수는 없긴 함.
        A_sing = np.clip(A_sing, -1.0, 1.0)
        B_sing = np.clip(B_sing, -1.0, 1.0)

        w_ang = np.arccos(S_Q / (A_sing.T @ B_sing))

    return w_ang

# %%
# 인접한 trial type pair에 대해 tangent space angle 계산
def compute_tangent_angle(rate_sorted, rate_sorted_mean_coll, mean_dist_mat_asis):
    
    n_neighbors = 5 # knn

    # 각 trial type에 대해 인접한 trial type과의 tangent space angle 계산
    list_angles2 = np.zeros(rate_sorted_mean_coll.shape[1]) # angle 하나만 기록할 때
    for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):

        # print(f'trial_type_ind = {trial_type_ind}')
        
        # 1. 가장 가까운 trial type 판정
        bool_not_tt = rate_sorted_mean_coll.columns != trial_type
        # adj_tt_ind = np.argmax(list_RSM_cos_coll[ind][trial_type_ind, bool_not_tt]) # 자기 자신은 뺌
        adj_tt_ind = np.argmin(mean_dist_mat_asis[trial_type_ind, bool_not_tt]) # argmin or argmax 주의!

        if adj_tt_ind >= trial_type_ind:
            adj_tt_ind = adj_tt_ind + 1 # tt를 제외하기 전 원래 index로 수정
        adj_tt = rate_sorted_mean_coll.columns[adj_tt_ind]

        # 2. orthogonal distance 계산

        rate_pair = rate_sorted.loc[:, [trial_type, adj_tt]].copy()
        rate_sorted_mean_coll_pair = rate_sorted_mean_coll.loc[:, [trial_type, adj_tt]].copy()

        # trial type별 mean point들 concatenate
        rate_plus_mean = pd.concat([rate_pair, rate_sorted_mean_coll_pair], axis=1)

        # geodesic distance matrix 계산
        n_components = 1 # 목표 차원 수
        # n_components = rate.shape[0] # 목표 차원 수
        n_neighbors = 5 # 이웃 점 개수

        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        
        rate_plus_mean_isomap = pd.DataFrame(isomap.fit_transform(rate_plus_mean.T).T, columns=rate_plus_mean.columns)
        dist_mat_asis_pair = isomap.dist_matrix_[:rate_pair.shape[1], rate_pair.shape[1]:].copy() # 각 점과 두 mean point의 거리 (num_neurons x 2)

        # 해당 trial type
        mat_orth_adj, _ = compute_orth_par_dist(trial_type, adj_tt, rate_pair, rate_sorted_mean_coll_pair)
        mat_orth_adj = pd.DataFrame(mat_orth_adj, index=rate_pair.index)
        mean_vector = rate_sorted_mean_coll.loc[:, adj_tt] - rate_sorted_mean_coll.loc[:, trial_type]
        radius_tt = np.percentile(np.linalg.norm(mat_orth_adj, axis=0), 90)

        # 인접한 trial type
        mat_orth_adj_rev, _ = compute_orth_par_dist(adj_tt, trial_type, rate_pair, rate_sorted_mean_coll_pair)
        mat_orth_adj_rev = pd.DataFrame(mat_orth_adj_rev, index=rate_pair.index)
        mean_vector_rev = rate_sorted_mean_coll.loc[:, trial_type] - rate_sorted_mean_coll.loc[:, adj_tt]
        radius_tt_rev = np.percentile(np.linalg.norm(mat_orth_adj_rev, axis=0), 90)

        rad_ratio_tt, rad_ratio_tt_rev = radius_tt / (radius_tt + radius_tt_rev), radius_tt_rev / (radius_tt + radius_tt_rev)
        bool_pos = np.array(mat_orth_adj.apply(lambda x : np.dot(x, rad_ratio_tt*mean_vector), axis=0) < np.linalg.norm(rad_ratio_tt*mean_vector)**2)
        bool_pos_rev = np.array(mat_orth_adj_rev.apply(lambda x : np.dot(x, rad_ratio_tt_rev*mean_vector_rev), axis=0) < np.linalg.norm(rad_ratio_tt_rev*mean_vector_rev)**2)

        # 3. inter-mean vector와 가장 각도 크고 & orthogonal distance가 mean point들의 중간쯤에 있는 trial들 선정
        
        # 해당 trial type
        rate_tt_pos = rate_pair.loc[:, trial_type].loc[:, bool_pos].copy()
        geo_tt = dist_mat_asis_pair[:rate_pair.loc[:, trial_type].shape[1]][bool_pos, 0].copy()
        cand_inds1 = np.array(geo_tt >= np.percentile(geo_tt, 75)) # mean point로부터 가장 먼 trial들
        dif_orth_tt = np.linalg.norm(mat_orth_adj.loc[:, bool_pos].sub(rad_ratio_tt*mean_vector, axis=0), axis=0)
        cand_inds2 = np.array(dif_orth_tt <= np.percentile(dif_orth_tt, 25)) # orthogonal distance가 mean point들의 중간쯤에 있는 trial들

        cand_inds = np.logical_and(cand_inds1, cand_inds2)
        cand_points_tt = rate_tt_pos.loc[:, cand_inds].copy()
        # print(cand_points_tt.shape[1])

        # 인접한 trial type
        rate_tt_pos_rev = rate_pair.loc[:, adj_tt].loc[:, bool_pos_rev].copy()
        geo_tt_rev = dist_mat_asis_pair[rate_pair.loc[:, trial_type].shape[1]:][bool_pos_rev, 1].copy()
        cand_inds1_rev = np.array(geo_tt_rev >= np.percentile(geo_tt_rev, 75)) # mean point로부터 가장 먼 trial들
        dif_orth_tt_rev = np.linalg.norm(mat_orth_adj_rev.loc[:, bool_pos_rev].sub(rad_ratio_tt_rev*mean_vector_rev, axis=0), axis=0)
        cand_inds2_rev = np.array(dif_orth_tt_rev <= np.percentile(dif_orth_tt_rev, 25)) # orthogonal distance가 mean point들의 중간쯤에 있는 trial들

        cand_inds_rev = np.logical_and(cand_inds1_rev, cand_inds2_rev)
        cand_points_tt_rev = rate_tt_pos_rev.loc[:, cand_inds_rev].copy()
        # print(cand_points_tt_rev.shape[1])

        # candidate point들 중 가까운 pair들의 index
        try:
            list_dist_cands = cdist(cand_points_tt.T, cand_points_tt_rev.T, 'euclidean')
            bool_close_pairs = list_dist_cands <= np.percentile(list_dist_cands, 50)
            list_close_pair_inds = np.argwhere(bool_close_pairs)
        except:
            # print(cand_points_tt, cand_points_tt_rev, sep='\n')
            # print(list_dist_cands.shape)
            list_close_pair_inds = []

        # 4. tangent space angle 계산 (improved tangent space approximation은 Zhang, 2011 참고)
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        
        rate_pair_pos = pd.concat([rate_tt_pos, rate_tt_pos_rev], axis=1)
        nbrs.fit(rate_pair_pos.T)
        nbr_inds = nbrs.kneighbors(rate_pair_pos.T, return_distance=False) # num_pair_trials x n_neighbors (neighbor는 본인 포함!)
        nbr_inds = pd.DataFrame(nbr_inds, index=rate_pair_pos.columns)
        nbr_inds_tt = nbr_inds.loc[trial_type].loc[cand_inds].copy()
        nbr_inds_tt_rev = nbr_inds.loc[adj_tt].loc[cand_inds_rev].copy()

        # 4-1. 해당 trial type
        list_PC_tt = np.zeros((cand_points_tt.shape[1], rate_sorted.shape[0], rate_sorted.shape[0])) # principal angles용
        list_PC_tt = np.empty(cand_points_tt.shape[1], dtype=object) # weighted angle용
        for cand_point_ind in range(cand_points_tt.shape[1]):
            cand_nbrs = rate_pair_pos.iloc[:, nbr_inds_tt.iloc[cand_point_ind]].copy()
            U, S, VT = np.linalg.svd(cand_nbrs.sub(cand_nbrs.mean(axis=1), axis=0), full_matrices=False) # PCA + weighted angle
            
            list_PC_tt[cand_point_ind] = dc([U, np.diag(S)]) # weighted angle용

        # 4-2. 인접한 trial type
        list_PC_tt_rev = np.zeros((cand_points_tt_rev.shape[1], rate_sorted.shape[0], rate_sorted.shape[0])) # principal angles용
        list_PC_tt_rev = np.empty(cand_points_tt_rev.shape[1], dtype=object) # weighted angle용
        for cand_point_ind in range(cand_points_tt_rev.shape[1]):
            cand_nbrs = rate_pair_pos.iloc[:, nbr_inds_tt_rev.iloc[cand_point_ind]].copy()
            U, S, VT = np.linalg.svd(cand_nbrs.sub(cand_nbrs.mean(axis=1), axis=0), full_matrices=False) # PCA + weighted angle
            
            list_PC_tt_rev[cand_point_ind] = dc([U, np.diag(S)]) # weighted angle용

        # candidate point pair 중 가까운 pair에 한정해서 principal angle 계산 (두 trial type 간 angle)
        list_angles = np.full(len(list_close_pair_inds), np.nan)
        for ind, close_pair_ind in enumerate(list_close_pair_inds):
            try:
                cand1, cand2 = close_pair_ind # euclidean 기준
                list_angles[ind] = weighted_angle(list_PC_tt[cand1][0], list_PC_tt_rev[cand2][0],
                                                    list_PC_tt[cand1][1], list_PC_tt_rev[cand2][1]) # 두 trial type과 concat space 간 각도 평균 # isomap 5d 이하일 땐 tangent니까 한 차원 빼서!
            except:
                list_angles[ind] = np.nan

        list_angles2[trial_type_ind] = np.nanmean(list_angles)

    return list_angles2

# %%
# local tangent space의 angle 측정

def compute_tangent_angle_sess(sess_ind):

    # if sess_ind != 0 and sess_ind != 6:

        c_proc = mp.current_process()
        print("Running on Process", c_proc.name, "PID", c_proc.pid)

        list_target_slopes = np.linspace(0, 2, 21, endpoint=True)
        # list_target_slopes = [0, 1, 2]

        num_trial_types = 119

        print(f'sess_ind: {sess_ind}')
        
        rate_sorted = list_rate_all[sess_ind].sort_index(axis=1)
        stm = rate_sorted.columns.copy()

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

        # tangent angle 계산
        start_t = time.time()
        list_angles_asis = compute_tangent_angle(rate_sorted, rate_sorted_mean_coll, mean_dist_mat_asis)
        print(f'as-is duration {time.time() - start_t:.0f} sec')

        # 평균이 0인 경우 NaN으로 바꾸기 (mean이 0인 경우와 var이 0인 경우가 정확히 일치하는 것을 이미 확인함.)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        n_neighbors = 5 # knn
        list_angles_RRneuron2 = np.zeros((len(list_target_slopes), num_trial_types)) # angle 하나만 기록할 때
        for slope_ind, target_slope in enumerate(list_target_slopes):
            print(f'target_slope = {target_slope:.1f}')
            
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

            # tangent angle 계산
            start_t = time.time()
            list_angles_RRneuron2[slope_ind] = compute_tangent_angle(rate_RRneuron_dr, rate_mean_RRneuron_coll, mean_dist_mat_asis)
            print(f'slope {target_slope:.1f} duration {time.time() - start_t:.0f} sec')

        # 파일에 저장
        filename = 'tangent_angles_ABO_' + str(sess_ind) + '.pickle'
        with open(filename, "wb") as f:
            pickle.dump({'tree_variables': ['list_angles_asis', 'list_angles_RRneuron2'],
                        'list_angles_asis': list_angles_asis, 'list_angles_RRneuron2': list_angles_RRneuron2}, f)

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
num_sess = len(list_rate_all)

# ABO tangent angle
if __name__ == '__main__':

    with mp.Pool(processes=12) as pool:
        list_inputs = [[sess_ind] for sess_ind in range(num_sess)]
        
        pool.starmap(compute_tangent_angle_sess, list_inputs)
