import numpy as np
import pandas as pd
from itertools import combinations
import math

from network_params import net_dict

n_clusters = net_dict['n_clusters']
n_stim_clusters = 1
n_stims = 6
n_trials = 50
dur_stim = 250 # ms
stim_amp = 0.15
# np.random.seed(0)
rng = np.random.default_rng(0)

# one set of clusters
stim_clusters = rng.choice(range(n_clusters), n_stim_clusters, replace=False)

# multiple sets of clusters
# pot_stim_clusters = np.array(list(combinations(range(n_clusters), n_stim_clusters))) # potential sets of all possible stimuli
# n_stims = np.min([n_stims, len(pot_stim_clusters)])
# ind_stim_clusters = rng.choice(range(len(pot_stim_clusters)), n_stims, replace=False)
# list_stim_clusters = pot_stim_clusters[ind_stim_clusters].copy() # stimuli to be used

# reduce computation cost
n_stims = np.min([n_stims, math.comb(n_clusters, n_stim_clusters)])
list_stim_clusters = np.full((n_stims, n_stim_clusters), np.nan)
stim_ind = 0
while np.isnan(list_stim_clusters[:, 0]).sum() > 0:
    cand_stim = rng.choice(range(n_clusters), n_stim_clusters, replace=False)
    if cand_stim not in list_stim_clusters:
        list_stim_clusters[stim_ind] = cand_stim.copy()
        stim_ind += 1
list_stim_clusters = list_stim_clusters.astype(int)

multi_stim_clusters = list_stim_clusters.copy()
stim_inds_trial = rng.permutation(np.repeat(range(n_stims), n_trials)) # randomize trial order

multi_stim_amps = [[stim_amp, 0.0] * n_trials] * n_stims
stim_inds_startend = np.lib.stride_tricks.sliding_window_view(np.arange(0, n_stims*n_trials+1), window_shape=2)
multi_stim_times = (stim_inds_startend * dur_stim).astype(np.float32)
multi_stim_times[1:, 0] += 0.1 # stim times must be strictly increasing, so add a little value to avoid overlap
multi_stim_times = [multi_stim_times[stim_inds_trial == stim_ind].flatten() for stim_ind in range(n_stims)]

stim_dict = {
    # list of clusters to be stimulated (None: no stimulation, 0-n_clusters-1)
    # "stim_clusters": stim_clusters,
    "multi_stim_clusters": multi_stim_clusters,
    "stim_inds_trial": stim_inds_trial,
    # stimulus amplitude (in pA)
    "stim_amp": 0.15,
    "multi_stim_amps": multi_stim_amps,
    # stimulus start times in ms: list (warmup time is added automatically)
    "stim_starts": [500],
    # list of stimulus end times in ms (warmup time is added automatically)
    "stim_ends": [1500],
    "multi_stim_times": multi_stim_times
}