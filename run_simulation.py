import matplotlib.pyplot as plt
import network
from helper import raster_plot
from network_params import net_dict
from sim_params import sim_dict
from stimulus_params import stim_dict

import pickle
from time import time
import multiprocessing as mp
from copy import deepcopy as dc
import numpy as np
import pandas as pd
from itertools import combinations, product
import math

def simulate_and_save(net_params=None, stim_dict_temp=None, stim_params=None, sim_params=None, misc_params=None):

    # Creates object which creates the EI clustered network in NEST
    net_dict_temp = dc(net_dict)
    sim_dict_temp = dc(sim_dict)
    save_name_sum = ''
    if net_params is not None:
        for key, value in net_params.items():
            net_dict_temp[key] = value
            if key != 'baseline_conn_prob':
                # if key not in ['I_th_E', 'I_th_I', 'N_E', 'N_I', 'n_clusters', 'rj', 'gei']:
                #     save_name_sum += '_' + key + f'{value:.1f}'
                # elif key in ['I_th_E', 'I_th_I', 'rj', 'gei']:
                #     save_name_sum += '_' + key + f'{value:.2f}'
                # else:
                #     save_name_sum += '_' + key + f'{value:.0f}'
                save_name_sum += '_' + key + str(value)
            else:
                save_name_sum += '_' + 'p'
                for p in value.flatten():
                    save_name_sum += str(int(p*10))
    if stim_dict_temp is None: # stim_dict default
        stim_dict_temp = dc(stim_dict)
    elif stim_dict_temp is not None and stim_params is not None: # if stim_dict was changed from default
        sim_dict_temp['simtime'] = float(stim_params['n_stims'] * stim_params['n_trials'] * stim_params['dur_stim'])
        for key, value in stim_params.items():
            # if key in ['n_stim_clusters', 'stim_amp']:
            # if key in ['n_stim_clusters', 'n_stims', 'n_trials', 'stim_amp']:
            # if key == 'stim_amp' and isinstance(value, np.ndarray):
            #     save_name_sum += '_stim_amp_het'
            # else:    
                save_name_sum += '_' + key + str(value)
            # elif key == 'stim_amp':
            #     save_name_sum += '_' + key + f'{value:.2f}'
    if sim_params is not None: # if sim_params was changed from default
        for key, value in sim_params.items():
            sim_dict_temp[key] = value
            save_name_sum += '_' + key + str(value)
    if misc_params is not None: # if misc_params was used
        for key, value in misc_params.items():
            save_name_sum += '_' + key + str(value)

    ei_network = network.ClusteredNetwork(sim_dict_temp, net_dict_temp, stim_dict_temp)
    start_time = time()

    # Runs the simulation and returns the spiketimes
    # get simulation initializes the network in NEST
    # and runs the simulation
    # it returns a dict with the average rates,
    # the spiketimes and the used parameters
    result = ei_network.get_simulation()
    if stim_params is not None:
        result.update({'stim_params': stim_params})
    # print(result['spiketimes'][0])
    ax = raster_plot(
        result["spiketimes"],
        tlim=(0, sim_dict_temp["simtime"]),
        colorgroups=[
            ("k", 0, net_dict_temp["N_E"]),
            ("darkred", net_dict_temp["N_E"], net_dict_temp["N_E"] + net_dict_temp["N_I"]),
        ],
    )
    # plt.savefig("clustered_ei_raster.png")
    plt.close()
    print(f"Firing rate of excitatory neurons: {result['e_rate']:6.2f} spikes/s")
    print(f"Firing rate of inhibitory neurons: {result['i_rate']:6.2f} spikes/s")
    print(f'Simulation required {(time()-start_time)/60:.2f} min')

    save_file_name = 'ei_clust_result'
    save_file_name = save_file_name + save_name_sum + '.pickle'
    with open(save_file_name, 'wb') as f:
        pickle.dump(result, f)

def create_stim_dict(n_clusters=6, n_stim_clusters=None, n_stim_neurons=None, n_stims=6, n_trials=50, dur_stim=250, stim_amp=0.15, randseed=0,
                     stim_overlap=True, match_amp=True, N_E=1200, gen_name=None):
    stim_dict = {
        # list of clusters to be stimulated (None: no stimulation, 0-n_clusters-1)
        # "stim_clusters": stim_clusters,
        # "multi_stim_clusters": multi_stim_clusters,
        # "stim_inds_trial": stim_inds_trial,
        # stimulus amplitude (in pA)
        "stim_amp": 0.15,
        # "multi_stim_amps": multi_stim_amps,
        # stimulus start times in ms: list (warmup time is added automatically)
        "stim_starts": [500],
        # list of stimulus end times in ms (warmup time is added automatically)
        "stim_ends": [1500],
        # "multi_stim_times": multi_stim_times
    }
    if n_stim_clusters is not None:
        stim_params={'n_stim_clusters': n_stim_clusters}
    elif n_stim_neurons is not None:
        stim_params={'n_stim_neurons': n_stim_neurons}
    else:
        raise Exception('Either n_stim_clusters or n_stim_neurons should not be None')
    new_keys = list(stim_params.keys()) + ['n_stims', 'n_trials', 'dur_stim', 'stim_amp', 'stim_overlap', 'match_amp']
    new_values = list(stim_params.values()) + [n_stims, n_trials, dur_stim, stim_amp, stim_overlap, match_amp]
    stim_params = dict(zip(new_keys, new_values))
    if gen_name is not None:
        stim_params['gen_name'] = gen_name
        stim_dict['gen_name'] = gen_name
    # stim_params={'n_stim_clusters': n_stim_clusters, 'n_stims': n_stims, 'n_trials': n_trials, 'dur_stim': dur_stim, 'stim_amp': stim_amp,
    #              'stim_overlap': stim_overlap, 'match_amp': match_amp, 'stim_ind_neu': stim_ind_neu, 'n_stim_neurons': n_stim_neurons}

    # np.random.seed(randseed)
    rng = np.random.default_rng(randseed)
    if n_stim_clusters is not None and n_stim_clusters != 0: # stimulate cluster by cluster
        # multiple sets of clusters
        if stim_overlap:
            raise Exception('Not determined')
        else:
            n_stims = np.min([n_stims, math.comb(n_clusters, n_stim_clusters)])
            assert n_clusters % n_stims == 0, 'Number of stimuli does not divide number of clusters'
            list_stim_clusters = rng.permutation(np.arange(n_clusters)).reshape(n_stims, -1)
        
        multi_stim_clusters = list_stim_clusters.copy()
        stim_inds_trial = rng.permutation(np.repeat(range(n_stims), n_trials)) # randomize trial order

        stim_inds_startend = np.lib.stride_tricks.sliding_window_view(np.arange(0, n_stims*n_trials+1), window_shape=2)
        multi_stim_times_temp = (stim_inds_startend * dur_stim).astype(np.float32)
        multi_stim_times_temp[1:, 0] += 0.1 # stim times must be strictly increasing, so add a little value to avoid overlap
        
        if stim_overlap:
            raise Exception('Not determined')
        else:
            multi_stim_amps = [[stim_amp, 0.0] * n_trials] * n_stims
            multi_stim_times = [multi_stim_times_temp[stim_inds_trial == stim_ind].flatten() for stim_ind in range(n_stims)]

        stim_dict['multi_stim_clusters'] = multi_stim_clusters.copy()
        stim_dict['stim_inds_trial'] = stim_inds_trial.copy()
        stim_dict['multi_stim_amps'] = dc(multi_stim_amps)
        stim_dict['multi_stim_times'] = multi_stim_times.copy()
    
    elif n_stim_neurons is not None and n_stim_neurons != 0: # stimulate individual neurons within clusters
        if stim_overlap:
            raise Exception('Not determined')
        else:
            if match_amp:
                if n_clusters == 1:
                    list_stim_neurons = np.arange(N_E).reshape(n_stims, n_stim_neurons)
                    list_stim_neurons_ind = np.empty(0)
                else:
                    raise Exception('Not determined')
            else:
                raise Exception('Not determined')
        
        multi_stim_neurons = list_stim_neurons.copy()
        stim_inds_trial = rng.permutation(np.repeat(range(n_stims), n_trials)) # randomize trial order

        # stim_amp
        if match_amp:
            multi_stim_amps = [[stim_amp, 0.0] * n_trials] * n_stims
        else:
            raise Exception('Not determined')

        # stim_times
        stim_inds_startend = np.lib.stride_tricks.sliding_window_view(np.arange(0, n_stims*n_trials+1), window_shape=2)
        multi_stim_times_temp = (stim_inds_startend * dur_stim).astype(np.float32)
        multi_stim_times_temp[1:, 0] += 0.1 # stim times must be strictly increasing, so add a little value (0.1 ms) to avoid overlap
        if stim_overlap:
            raise Exception('Not determined')
        else:
            multi_stim_times = [multi_stim_times_temp[stim_inds_trial == stim_ind].flatten() for stim_ind in range(n_stims)]

        stim_dict['multi_stim_neurons'] = multi_stim_neurons.copy()
        stim_dict['stim_inds_trial'] = stim_inds_trial.copy()
        stim_dict['multi_stim_amps'] = dc(multi_stim_amps)
        stim_dict['multi_stim_times'] = multi_stim_times.copy()
        stim_dict['list_stim_neurons_ind'] = list_stim_neurons_ind.copy()

    return stim_dict, stim_params

# %%
# simulate
if __name__ == "__main__":

    # when Q is not 1
    list_Q = [5, 10, 20, 50, 100]    
    n_sessions = 10
    list_seed = np.linspace(1, 901, n_sessions, endpoint=True).astype(int)
    N_E = 1200
    rep = 'm4'
    for seed in list_seed:
        for Q in list_Q:
            print(f'seed = {seed}, Q = {Q}')
            net_params = {'rep': rep, 'n_clusters': Q, 'conn_seed': seed}
            stim_dict_temp, stim_params = create_stim_dict(n_clusters=Q, n_stim_clusters=int(Q*0.2), n_stims=5, n_trials=200, stim_amp=0.1,
                                                           stim_overlap=False, match_amp=True)
            misc_params = {'adjmat': True}
            simulate_and_save(net_params=net_params, stim_dict_temp=stim_dict_temp, stim_params=stim_params, misc_params=misc_params)

    # Q = 1
    list_Q = [1]
    n_sessions = 10
    list_seed = np.linspace(1, 901, n_sessions, endpoint=True).astype(int)
    N_E = 1200
    rep = 'm4'
    for seed in list_seed:
        for Q in list_Q:
            print(f'seed = {seed}, Q = {Q}')
            net_params = {'n_clusters': Q, 'conn_seed': seed}
            stim_dict_temp, stim_params = create_stim_dict(n_clusters=Q, n_stim_neurons=int(N_E*0.2), n_stims=5, n_trials=200, stim_amp=0.1,
                                                           stim_overlap=False, match_amp=True)
            misc_params = {'adjmat': True}
            simulate_and_save(net_params=net_params, stim_dict_temp=stim_dict_temp, stim_params=stim_params, misc_params=misc_params)
