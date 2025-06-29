# %%
import mat73
import h5py
import hdf5storage as st
# from pymatreader import read_mat
import pickle

import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wilcoxon, norm, kruskal, tukey_hsd, poisson, nbinom
from scipy.io import loadmat
from scipy.special import gamma, gammaln
import scipy.optimize as opt
import seaborn as sns
from copy import deepcopy as dc
from statsmodels.stats.multitest import multipletests

from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, KFold, StratifiedKFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import time

# from libsvm import svmutil

# import torch
# import torch.nn as nn
# import torch.optim as optim

# from pycaret.classification import *

# %%
# negative log likelihood 함수 (modulated Poisson)
# def neg_log_likelihood_mod1(list_mu_var_G, spike_counts):
def neg_log_likelihood_mod1(list_s_r, spike_counts):
    r = list_s_r[1]
    s = list_s_r[0]
    # var_G = list_mu_var_G[-1]
    # mu = list_mu_var_G[0]

    # if mu > 0 and var_G > 0:
    # LL = gammaln(spike_counts + 1/var_G) - gammaln(spike_counts + 1) - gammaln(1/var_G)\
    #     + spike_counts * np.log(var_G*mu) - (spike_counts + 1/var_G) * np.log(var_G*mu + 1)
    LL = gammaln(spike_counts + r) - gammaln(spike_counts + 1) - gammaln(r)\
        - r * np.log(1 + s) + spike_counts * (np.log(s) - np.log(1 + s))
    nLL = np.sum(-LL)
    # else:
    #     nLL = np.nan

    return nLL

# %%
def fit_test_mod1(neu_ind, rate_train, rate_test, label_cnt_dict_train, rate_mean_train):
    
    ''' 각 trial type별로 parameter fitting '''

    list_nLL_mod_tt = np.full(len(label_cnt_dict_train), np.nan)
    list_var_G_estim = np.full(len(label_cnt_dict_train), np.nan)
    list_mu_estim = np.full(len(label_cnt_dict_train), np.nan)

    for trial_type_ind, trial_type_train in enumerate(label_cnt_dict_train):
        
        # train data에서 특정 trial type 내의 모든 trial에 대해 rate 0인지 기록
        all_zero = (rate_train.loc[neu_ind, trial_type_train].sum() == 0)
                
        if ~all_zero:
            # likelihood function에 필요한 변수들 선언
            spike_counts = rate_train.loc[neu_ind, trial_type_train].copy()
            initial_params = np.array([rate_mean_train.loc[neu_ind, trial_type_train], 1])
            
            # nLL (negative log likelihood) minimization
            res = opt.minimize(fun=neg_log_likelihood_mod1, x0=initial_params, args=(spike_counts), \
                method='Nelder-Mead', bounds=[(10**(-3), None), (10**(-3), None)])
            list_var_G_estim[trial_type_ind] = 1 / res.x[1]
            # list_var_G_estim[trial_type_ind] = res.x[1]
            list_mu_estim[trial_type_ind] = res.x[0] * res.x[1]
            # list_mu_estim[trial_type_ind] = res.x[0]

            # test nLL 계산
            if rate_test is not None:
                spike_counts = rate_test.loc[neu_ind, trial_type_train].copy()
                nLL_test_mod = neg_log_likelihood_mod1(res.x, spike_counts) # res.x는 fitting된 parameter들
                list_nLL_mod_tt[trial_type_ind] = nLL_test_mod

    return list_nLL_mod_tt, list_var_G_estim, list_mu_estim

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
def compute_mean_var_trial_collapse_cv(label_cnt_dict_train, label_cnt_dict_test, rate_train_sorted, rate_test_sorted):    
    list_trial_mean_train = [[0]] * len(label_cnt_dict_train)
    list_trial_var_train = [[0]] * len(label_cnt_dict_train)
    list_trial_mean_test = [[0]] * len(label_cnt_dict_train)
    list_trial_var_test = [[0]] * len(label_cnt_dict_train)

    for trial_type_ind, (trial_type_train, trial_type_test) in enumerate(zip(label_cnt_dict_train, label_cnt_dict_test)):
        
        # train data
        trial_rate_train = np.array(rate_train_sorted.loc[trial_type_train])                
        trial_mean_train = np.mean(trial_rate_train, axis=0, dtype=np.longdouble)
        trial_var_train = np.var(trial_rate_train, axis=0, ddof=1, dtype=np.longdouble)

        trial_mean_train = pd.DataFrame(trial_mean_train, columns=[trial_type_train], index=rate_train_sorted.columns)
        trial_var_train = pd.DataFrame(trial_var_train, columns=[trial_type_train], index=rate_train_sorted.columns)
        list_trial_mean_train[trial_type_ind] = trial_mean_train
        list_trial_var_train[trial_type_ind] = trial_var_train
            
        # test data
        trial_rate_test = np.array(rate_test_sorted.loc[trial_type_test])                
        trial_mean_test = np.mean(trial_rate_test, axis=0, dtype=np.longdouble)
        trial_var_test = np.var(trial_rate_test, axis=0, ddof=1, dtype=np.longdouble)

        trial_mean_test = pd.DataFrame(trial_mean_test, columns=[trial_type_test], index=rate_test_sorted.columns)
        trial_var_test = pd.DataFrame(trial_var_test, columns=[trial_type_test], index=rate_test_sorted.columns)                
        list_trial_mean_test[trial_type_ind] = trial_mean_test
        list_trial_var_test[trial_type_ind] = trial_var_test

    return list_trial_mean_train, list_trial_var_train, list_trial_mean_test, list_trial_var_test

# %%
def compare_two_poissons(sess_ind):

    ''' 각 trial type, 각 neuron별로 r, s fitting. 따라서 across trial types, across neurons 모두 가능 '''

    c_proc = mp.current_process()
    print("Running on Process",c_proc.name,"PID",c_proc.pid)

    # Modulated Poisson vs. Ordinary Poisson Goodness-of-Fit 비교 (all neurons)

    print(f'session index: {sess_ind}')
    
    rate = dc(list_rate_all[sess_ind])
    stm = rate.columns.copy()

    # delta t 곱해서 spike count로 만들기
    rate = rate * 0.25

    all_stm_unique, all_stm_counts = np.unique(stm, return_counts=True) # 모든 trial type counting
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    # parameter fitting

    # stm type별 counting & string name dictionary 제작
    rate_sorted = rate.sort_index(axis=1)
    stm_sorted = np.array(sorted(stm))

    all_stm_unique, all_stm_counts = np.unique(stm_sorted, return_counts=True) # 모든 trial type counting
    stm_cnt_dict = dict(zip(all_stm_unique, all_stm_counts))

    rate_mean, _ = compute_mean_var_trial_collapse(stm_cnt_dict, rate_sorted)
    sample_mean = dc(rate_mean)

    # MLE (maximum likelihood estimation)로 뉴런마다 parameter fitting
    
    list_var_G_estim_neu = np.zeros((rate_sorted.shape[0], all_stm_unique.shape[0]))
    list_mu_estim_neu = np.zeros((rate_sorted.shape[0], all_stm_unique.shape[0]))
    
    start = time.time()
    for neu_ind in rate_sorted.index:
        
        # modulated Poisson
        _, list_var_G_estim, list_mu_estim = \
            fit_test_mod1(neu_ind, rate_sorted, None, stm_cnt_dict, rate_mean)
        
        list_var_G_estim_neu[neu_ind] = dc(list_var_G_estim)
        list_mu_estim_neu[neu_ind] = dc(list_mu_estim)

        if neu_ind % 5 == 0:
            print(f'neu_ind2: {neu_ind} / {rate_sorted.shape[0]}, duration: {time.time() - start:.2f} sec')
            start = time.time()

    # 변수 파일에 저장
    filename = 'poisson_fit_rs_sep_ABO_' + str(sess_ind) + '.pickle'
    with open(filename, "wb") as f:
        # pickle.dump({'tree_variables': ['list_var_G_estim_neu', 'list_nLL_mod_cv', 'list_mu_estim_neu', 'sample_mean'],
        #              'list_var_G_estim_neu': list_var_G_estim_neu, 'list_nLL_mod_cv': list_nLL_mod_cv, 'list_mu_estim_neu': list_mu_estim_neu, 'sample_mean': sample_mean}, f)
        pickle.dump({'tree_variables': ['list_var_G_estim_neu', 'list_mu_estim_neu', 'sample_mean'],
                     'list_var_G_estim_neu': list_var_G_estim_neu, 'list_mu_estim_neu': list_mu_estim_neu, 'sample_mean': sample_mean}, f)
            
    print("Ended Process",c_proc.name)

# %%
# 변수 loading

# openscope
with open('SVM_prerequisite_variables.pickle', 'rb') as f:
    SVM_prerequisite_variables = pickle.load(f)
    
    list_rate_w1 = dc(SVM_prerequisite_variables['list_rate_w1'])
    list_stm_w1 = dc(SVM_prerequisite_variables['list_stm_w1'])
    list_neu_loc = dc(SVM_prerequisite_variables['list_neu_loc'])
    list_wfdur = dc(SVM_prerequisite_variables['list_wfdur'])
    list_slopes_an_loglog_12 = dc(SVM_prerequisite_variables['list_slopes_an_loglog_12']) # high repeat trial type 주의

# ABO Neuropixels
with open('resp_matrix_ep_RS_all_32sess_allensdk.pickle', 'rb') as f:
    resp_matrix_ep_RS_all = pickle.load(f)

    list_rate_RS = dc(resp_matrix_ep_RS_all['list_rate_RS'])
    list_rate_RS_dr = dc(resp_matrix_ep_RS_all['list_rate_RS_dr'])
    list_rate_all = dc(resp_matrix_ep_RS_all['list_rate_all'])
    list_rate_all_dr = dc(resp_matrix_ep_RS_all['list_rate_all_dr'])
    list_slopes_RS_an_loglog = dc(resp_matrix_ep_RS_all['list_slopes_RS_an_loglog'])
    list_slopes_all_an_loglog = dc(resp_matrix_ep_RS_all['list_slopes_all_an_loglog'])

    sess_inds_qual_all = dc(resp_matrix_ep_RS_all['sess_inds_qual_all'])

# %%
# multiprocessing

num_sess = 32
if __name__ == '__main__':
    
    with mp.Pool() as pool:    
        list_inputs = [[sess_ind] for sess_ind in range(num_sess)]
        
        pool.starmap(compare_two_poissons, list_inputs)
