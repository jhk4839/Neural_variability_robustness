import numpy as np

net_dict = {
    ############################################
    # neuron parameters
    ############################################
    # neuron model
    "neuron_type": "iaf_psc_exp",
    # Resting potential [mV]
    "E_L": 0.0,
    # Membrane capacitance [pF]
    "C_m": 1.0,
    # Membrane time constant for excitatory neurons [ms]
    "tau_E": 20.0,
    # Membrane time constant for inhibitory neurons [ms]
    "tau_I": 10.0,
    # Refractory period [ms]
    "t_ref": 5.0,
    # Threshold for excitatory neurons [mV]
    "V_th_E": 15.0,
    # Threshold for inhibitory neurons [mV]
    "V_th_I": 15.0,
    # Reset potential [mV]
    "V_r": 0.0,
    # synaptic time constant for excitatory synapses [ms]
    "tau_syn_ex": 3.0,
    # synaptic time constant for inhibitory synapses [ms]
    "tau_syn_in": 2.0,
    # synaptic delay [ms]
    "delay": 0.1,
    # Feed forward excitatory input [rheobase current]
    "I_th_E": 1.25,
    # Feed forward inhibitory input [rheobase current]
    "I_th_I": 0.78,
    # distribution of feed forward input,
    # I_th*[1-delta_I_../2, 1+delta_I_../2]
    "delta_I_xE": 0.0,  # excitatory
    "delta_I_xI": 0.0,  # inhibitory
    # initial membrane potential: either a float (in mV) to initialize all neurons to a fixed value
    # or "rand" for randomized values: "V_th_{E,I}" - 20 * nest.random.lognormal(0, 1)
    "V_m": "rand",
    ############################################
    # network parameters
    ############################################
    # number of excitatory neurons in the network
    # Neurons per cluster N_E/n_clusters
    "N_E": 1200,
    # number of inhibitory neurons in the network
    "N_I": 300,
    # Number of clusters
    "n_clusters": 6,
    # connection probabilities
    # baseline_conn_prob[0, 0] E to E, baseline_conn_prob[0, 1] I to E,
    # baseline_conn_prob[1, 0] E to I, baseline_conn_prob[1, 1] I to I
    "baseline_conn_prob": np.array([[0.2, 0.5], [0.5, 0.5]]),
    # inhibitory weight ratios - scaling like random balanced network
    "gei": 1.2,  # I to E
    "gie": 1.0,  # E to I
    "gii": 1.0,  # I to I
    # additional scaling factor for all weights
    # - can be used to scale weights with network size
    "s": 1.0,
    # fixed indegree - otherwise established with probability ps
    "fixed_indegree": False,
    # cluster network by "weight" or "probabilities"
    "clustering": "weight",
    # ratio excitatory to inhibitory clustering,
    # rj = 0 means no clustering, which resembles a clustered network
    # with a blanket of inhibition
    "rj": 0.75,
    # excitatory clustering factor,
    # rep = 1 means no clustering, reselmbles a balanced random network
    "rep": 3.2,
}