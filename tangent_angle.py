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
# Function to compute cosine similarity
def cos_sim(x, y):
    # x and y are 1D vectors

    # Remove NaN
    x, y = np.array(x), np.array(y)
    bool_notnan = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x, y = x[bool_notnan].copy(), y[bool_notnan].copy()

    return np.dot(x, y) / (np.linalg.norm(x.astype(np.float32)) * np.linalg.norm(y.astype(np.float32)))

# %%
# Function to compute orthogonal & parallel distance
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

    if B_left.shape[0] > 1: # more than 1 basis
        Q = (A_left@A_sing).T @ (B_left@B_sing)
        try:
            _, S_Q, _ = np.linalg.svd(Q.astype(np.float32))
        except:
            S_Q = np.full(np.min(Q.shape), np.nan)
        # print(f'singular values: {s}')
        # print(s[np.logical_or(s < -1, s > 1)])

        # Clamp singular values to the interval [-1, 1] to avoid numerical issues
        S_Q = np.clip(S_Q, -1.0, 1.0) 
        A_sing = np.clip(A_sing, -1.0, 1.0)
        B_sing = np.clip(B_sing, -1.0, 1.0)

        w_ang = np.arccos(np.sum(S_Q) / np.trace(A_sing.T @ B_sing))

    else: # # 1 basis
        S_Q = (A_left@A_sing).T @ (B_left@B_sing) # scalar

        # Clamp singular values to the interval [-1, 1] to avoid numerical issues
        S_Q = np.clip(S_Q, -1.0, 1.0) 
        A_sing = np.clip(A_sing, -1.0, 1.0)
        B_sing = np.clip(B_sing, -1.0, 1.0)

        w_ang = np.arccos(S_Q / (A_sing.T @ B_sing))

    return w_ang

# %%
# Compute tangent space angles for neighboring stimulus pairs
def compute_tangent_angle(rate_sorted, rate_sorted_mean_coll, mean_dist_mat_asis):

    # For each stimulus, compute tangent space angle with its neighbor
    list_angles2 = np.zeros(rate_sorted_mean_coll.shape[1]) 
    for trial_type_ind, trial_type in enumerate(rate_sorted_mean_coll.columns):

        # print(f'trial_type_ind = {trial_type_ind}')
        
        # 1. Determine the nearest neighbor stimulus
        bool_not_tt = rate_sorted_mean_coll.columns != trial_type
        # adj_tt_ind = np.argmax(list_RSM_cos_coll[ind][trial_type_ind, bool_not_tt]) # exclude current stimulus
        adj_tt_ind = np.argmin(mean_dist_mat_asis[trial_type_ind, bool_not_tt]) 

        if adj_tt_ind >= trial_type_ind:
            adj_tt_ind = adj_tt_ind + 1 # to original index before excluding the stimulus
        adj_tt = rate_sorted_mean_coll.columns[adj_tt_ind]

        # 2. Compute orthogonal distance

        rate_pair = rate_sorted.loc[:, [trial_type, adj_tt]].copy()
        rate_sorted_mean_coll_pair = rate_sorted_mean_coll.loc[:, [trial_type, adj_tt]].copy()

        # concatenate centroids of all stimuli
        rate_plus_mean = pd.concat([rate_pair, rate_sorted_mean_coll_pair], axis=1)

        # Compute geodesic distance matrix
        n_components = 1 # target number of dimensions
        # n_components = rate.shape[0] # target number of dimensions
        n_neighbors = 5 # number of neighbors

        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        
        rate_plus_mean_isomap = pd.DataFrame(isomap.fit_transform(rate_plus_mean.T).T, columns=rate_plus_mean.columns)
        dist_mat_asis_pair = isomap.dist_matrix_[:rate_pair.shape[1], rate_pair.shape[1]:].copy() # distance from the 2 centroids (num_neurons x 2)

        # current stimulus
        mat_orth_adj, _ = compute_orth_par_dist(trial_type, adj_tt, rate_pair, rate_sorted_mean_coll_pair)
        mat_orth_adj = pd.DataFrame(mat_orth_adj, index=rate_pair.index)
        mean_vector = rate_sorted_mean_coll.loc[:, adj_tt] - rate_sorted_mean_coll.loc[:, trial_type]
        radius_tt = np.percentile(np.linalg.norm(mat_orth_adj.astype(np.float32), axis=0), 90)

        # neighbor stimulus
        mat_orth_adj_rev, _ = compute_orth_par_dist(adj_tt, trial_type, rate_pair, rate_sorted_mean_coll_pair)
        mat_orth_adj_rev = pd.DataFrame(mat_orth_adj_rev, index=rate_pair.index)
        mean_vector_rev = rate_sorted_mean_coll.loc[:, trial_type] - rate_sorted_mean_coll.loc[:, adj_tt]
        radius_tt_rev = np.percentile(np.linalg.norm(mat_orth_adj_rev.astype(np.float32), axis=0), 90)

        rad_ratio_tt, rad_ratio_tt_rev = radius_tt / (radius_tt + radius_tt_rev), radius_tt_rev / (radius_tt + radius_tt_rev)
        bool_pos = np.array(mat_orth_adj.apply(lambda x : np.dot(x, rad_ratio_tt*mean_vector), axis=0) < np.linalg.norm((rad_ratio_tt*mean_vector).astype(np.float32))**2)
        bool_pos_rev = np.array(mat_orth_adj_rev.apply(lambda x : np.dot(x, rad_ratio_tt_rev*mean_vector_rev), axis=0) < np.linalg.norm((rad_ratio_tt_rev*mean_vector_rev).astype(np.float32))**2)

        # 3. Select trials farthest from centroid & having orthogonal distances near midpoint
        
        # current stimulus
        rate_tt_pos = rate_pair.loc[:, trial_type].loc[:, bool_pos].copy()
        geo_tt = dist_mat_asis_pair[:rate_pair.loc[:, trial_type].shape[1]][bool_pos, 0].copy()
        cand_inds1 = np.array(geo_tt >= np.percentile(geo_tt, 75)) # trials farthest from the centroid
        dif_orth_tt = np.linalg.norm(mat_orth_adj.loc[:, bool_pos].sub(rad_ratio_tt*mean_vector, axis=0).astype(np.float32), axis=0)
        cand_inds2 = np.array(dif_orth_tt <= np.percentile(dif_orth_tt, 25)) # trials with orthogonal distances near midpoint

        cand_inds = np.logical_and(cand_inds1, cand_inds2)
        cand_points_tt = rate_tt_pos.loc[:, cand_inds].copy()
        # print(cand_points_tt.shape[1])

        # neighbor stimulus
        rate_tt_pos_rev = rate_pair.loc[:, adj_tt].loc[:, bool_pos_rev].copy()
        geo_tt_rev = dist_mat_asis_pair[rate_pair.loc[:, trial_type].shape[1]:][bool_pos_rev, 1].copy()
        cand_inds1_rev = np.array(geo_tt_rev >= np.percentile(geo_tt_rev, 75)) # trials farthest from the centroid
        dif_orth_tt_rev = np.linalg.norm(mat_orth_adj_rev.loc[:, bool_pos_rev].sub(rad_ratio_tt_rev*mean_vector_rev, axis=0).astype(np.float32), axis=0)
        cand_inds2_rev = np.array(dif_orth_tt_rev <= np.percentile(dif_orth_tt_rev, 25)) # trials with orthogonal distances near midpoint

        cand_inds_rev = np.logical_and(cand_inds1_rev, cand_inds2_rev)
        cand_points_tt_rev = rate_tt_pos_rev.loc[:, cand_inds_rev].copy()
        # print(cand_points_tt_rev.shape[1])

        # Identify close candidate pairs
        try:
            list_dist_cands = cdist(cand_points_tt.T, cand_points_tt_rev.T, 'euclidean')
            bool_close_pairs = list_dist_cands <= np.percentile(list_dist_cands, 50)
            list_close_pair_inds = np.argwhere(bool_close_pairs)
        except:
            # print(cand_points_tt, cand_points_tt_rev, sep='\n')
            # print(list_dist_cands.shape)
            list_close_pair_inds = []

        # 4. Compute tangent space angles
        
        n_neighbors = 5 # knn
        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        
        rate_pair_pos = pd.concat([rate_tt_pos, rate_tt_pos_rev], axis=1)
        nbrs.fit(rate_pair_pos.T)
        nbr_inds = nbrs.kneighbors(rate_pair_pos.T, return_distance=False) # num_pair_trials x n_neighbors (neighbors can include the query)
        nbr_inds = pd.DataFrame(nbr_inds, index=rate_pair_pos.columns)
        nbr_inds_tt = nbr_inds.loc[trial_type].loc[cand_inds].copy()
        nbr_inds_tt_rev = nbr_inds.loc[adj_tt].loc[cand_inds_rev].copy()

        # 4-1. current stimulus
        list_PC_tt = np.zeros((cand_points_tt.shape[1], rate_sorted.shape[0], rate_sorted.shape[0])) 
        list_PC_tt = np.empty(cand_points_tt.shape[1], dtype=object) # weighted angle
        for cand_point_ind in range(cand_points_tt.shape[1]):
            cand_nbrs = rate_pair_pos.iloc[:, nbr_inds_tt.iloc[cand_point_ind]].copy()
            U, S, VT = np.linalg.svd(cand_nbrs.sub(cand_nbrs.mean(axis=1), axis=0).astype(np.float32), full_matrices=False) # PCA + weighted angle
            
            list_PC_tt[cand_point_ind] = dc([U, np.diag(S)]) # weighted angle

        # 4-2. neighbor stimulus
        list_PC_tt_rev = np.zeros((cand_points_tt_rev.shape[1], rate_sorted.shape[0], rate_sorted.shape[0])) 
        list_PC_tt_rev = np.empty(cand_points_tt_rev.shape[1], dtype=object) # weighted angle
        for cand_point_ind in range(cand_points_tt_rev.shape[1]):
            cand_nbrs = rate_pair_pos.iloc[:, nbr_inds_tt_rev.iloc[cand_point_ind]].copy()
            U, S, VT = np.linalg.svd(cand_nbrs.sub(cand_nbrs.mean(axis=1), axis=0).astype(np.float32), full_matrices=False) # PCA + weighted angle
            
            list_PC_tt_rev[cand_point_ind] = dc([U, np.diag(S)]) # weighted angle

        # Compute weighted angle for close candidate pairs
        list_angles = np.full(len(list_close_pair_inds), np.nan)
        for ind, close_pair_ind in enumerate(list_close_pair_inds):
            try:
                cand1, cand2 = close_pair_ind
                list_angles[ind] = weighted_angle(list_PC_tt[cand1][0], list_PC_tt_rev[cand2][0],
                                                    list_PC_tt[cand1][1], list_PC_tt_rev[cand2][1]) 
            except:
                list_angles[ind] = np.nan

        list_angles2[trial_type_ind] = np.nanmean(list_angles)

    return list_angles2

# %%
# Measure local tangent space angles per session

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

        # Multiply by delta t to convert to spike counts
        rate_sorted = rate_sorted * 0.25 

        # Create a counting dictionary for each stimulus
        all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) 
        stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

        # Compute mean & variance for each stimulus
        rate_sorted_mean, rate_sorted_var = compute_mean_var_trial(stm_cnt_dict, rate_sorted)
        rate_sorted_mean_coll, rate_sorted_var_coll = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)

        list_slopes_dr = pd.DataFrame(list_slopes_all_an_loglog[sess_ind], \
                                        columns=rate_sorted_mean_coll.columns).copy()

        # inter-centroid geodesic distance

        # concatenate centroids of all stimuli
        rate_plus_mean = pd.concat([rate_sorted, rate_sorted_mean_coll], axis=1)

        # Compute geodesic distance matrix
        n_components = 1 # target number of dimensions
        # n_components = rate.shape[0] # target number of dimensions
        n_neighbors = 5 # number of neighbors

        isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        
        isomap.fit(rate_plus_mean.T)
        mean_dist_mat_asis = isomap.dist_matrix_[rate_sorted.shape[1]:, rate_sorted.shape[1]:].copy() # inter-centroid geodesic distance matrix

        # Compute tangent angles
        start_t = time.time()
        list_angles_asis = compute_tangent_angle(rate_sorted, rate_sorted_mean_coll, mean_dist_mat_asis)
        print(f'as-is duration {time.time() - start_t:.0f} sec')

        # Convert 0 to NaN (verified that cases of mean=0 and var=0 coincide exactly)
        rate_sorted_mean_coll[rate_sorted_mean_coll == 0] = np.nan
        rate_sorted_var_coll[rate_sorted_var_coll == 0] = np.nan

        n_neighbors = 5 # knn
        list_angles_RRneuron2 = np.zeros((len(list_target_slopes), num_trial_types)) 
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

            # Compute tangent angles
            start_t = time.time()
            list_angles_RRneuron2[slope_ind] = compute_tangent_angle(rate_RRneuron_dr, rate_mean_RRneuron_coll, mean_dist_mat_asis)
            print(f'slope {target_slope:.1f} duration {time.time() - start_t:.0f} sec')

        # Save into a file
        filename = 'D:\\Users\\USER\\Shin Lab\\code\\tangent_angles_ABO_' + str(sess_ind) + '.pickle'
        with open(filename, "wb") as f:
            pickle.dump({'tree_variables': ['list_angles_asis', 'list_angles_RRneuron2'],
                        'list_angles_asis': list_angles_asis, 'list_angles_RRneuron2': list_angles_RRneuron2}, f)

        print("Ended Process", c_proc.name)

# %%
# loading variables

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

    with mp.Pool() as pool:
        list_inputs = [[sess_ind] for sess_ind in range(num_sess) if sess_ind == 5]
        
        pool.starmap(compute_tangent_angle_sess, list_inputs)
