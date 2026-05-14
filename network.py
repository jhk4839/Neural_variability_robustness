import pickle
# import os
from copy import deepcopy as dc

import helper
import nest
import numpy as np


class ClusteredNetwork:
    """EI-clustered network objeect to build and simulate the network.

    Provides functions to create neuron populations,
    stimulation devices and recording devices for an
    EI-clustered network and setups the simulation in
    NEST (v3.x).

    Attributes
    ----------
    _params: dict
        Dictionary with parameters used to construct network.
    _populations: list
        List of neuron population groups.
    _recording_devices: list
        List of recording devices.
    _currentsources: list
        List of current sources.
    _model_build_pipeline: list
        List of functions to build the network.
    """

    def __init__(self, sim_dict, net_dict, stim_dict):
    # def __init__(self, _dict):
        """Initialize the ClusteredNetwork object.

        Parameters are given and explained in the files network_params_EI.py,
        sim_params_EI.py and stimulus_params_EI.py.

        Parameters
        ----------
        sim_dict: dict
            Dictionary with simulation parameters.
        net_dict: dict
            Dictionary with network parameters.
        stim_dict: dict
            Dictionary with stimulus parameters.
        """

        # merge dictionaries of simulation, network and stimulus parameters
        self._params = {**sim_dict, **net_dict, **stim_dict}
        # self._params = dc(_dict)

        # list of neuron population groups [E_pops, I_pops]
        self._populations = []
        self._recording_devices = []
        self._currentsources = []
        self._model_build_pipeline = [
            self.setup_nest,
            self.create_populations,
            self.create_stimulation,
            self.create_recording_devices,
            self.connect,
        ]

        if self._params["clustering"] == "weight":
            if self._params["rep"] is not None and not isinstance(self._params["rep"], str): # revised from original version of Rostami et al., 2024
                jep = self._params["rep"]
                jip = 1.0 + (jep - 1) * self._params["rj"]
                self._params["jplus"] = np.array([[jep, jip], [jip, jip]])
        elif self._params["clustering"] == "probabilities":
            pep = self._params["rep"]
            pip = 1.0 + (pep - 1) ** self._params["rj"]
            self._params["pplus"] = np.array([[pep, pip], [pip, pip]])
        else:
            raise ValueError("Clustering type not recognized")

    def setup_nest(self):
        """Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        Updates randseed of parameters to the actual
        used one if none is supplied.
        """

        nest.ResetKernel()
        nest.set_verbosity("M_WARNING")
        nest.local_num_threads = self._params.get("n_vp", 4)
        nest.resolution = self._params.get("dt")
        self._params["randseed"] = self._params.get("randseed")
        nest.rng_seed = self._params.get("randseed")

    def create_populations(self):
        """Create all neuron populations.

        n_clusters excitatory and inhibitory neuron populations
        with the parameters of the network are created.
        """

        # make sure number of clusters and units are compatible
        # if self._params["N_E"] % self._params["n_clusters"] != 0:
        #     raise ValueError("N_E must be a multiple of Q")
        # if self._params["N_I"] % self._params["n_clusters"] != 0:
        #     raise ValueError("N_E must be a multiple of Q")
        if self._params["neuron_type"] != "iaf_psc_exp":
            raise ValueError("Model only implemented for iaf_psc_exp neuron model")

        if self._params["I_th_E"] is None:
            I_xE = self._params["I_xE"]  # I_xE is the feed forward excitatory input in pA
        else:
            if self._params.get('gen_name') is None: # revised from original version of Rostami et al., 2024
                I_xE = self._params["I_th_E"] * helper.rheobase_current(
                    self._params["tau_E"], self._params["E_L"], self._params["V_th_E"], self._params["C_m"]
                )
            else:
                I_xE = 0
        if self._params["I_th_I"] is None:
            I_xI = self._params["I_xI"]
        else:
            if self._params.get('gen_name') is None: # revised from original version of Rostami et al., 2024
                I_xI = self._params["I_th_I"] * helper.rheobase_current(
                    self._params["tau_I"], self._params["E_L"], self._params["V_th_I"], self._params["C_m"]
                )
            else:
                I_xI = 0

        E_neuron_params = {
            "E_L": self._params["E_L"],
            "C_m": self._params["C_m"],
            "tau_m": self._params["tau_E"],
            "t_ref": self._params["t_ref"],
            "V_th": self._params["V_th_E"],
            "V_reset": self._params["V_r"],
            "I_e": I_xE
            if self._params["delta_I_xE"] == 0
            else I_xE * nest.random.uniform(1 - self._params["delta_I_xE"] / 2, 1 + self._params["delta_I_xE"] / 2),
            "tau_syn_ex": self._params["tau_syn_ex"],
            "tau_syn_in": self._params["tau_syn_in"],
            "V_m": self._params["V_m"]
            if not self._params["V_m"] == "rand"
            else self._params["V_th_E"] - 20 * nest.random.lognormal(0, 1),
        }
        I_neuron_params = {
            "E_L": self._params["E_L"],
            "C_m": self._params["C_m"],
            "tau_m": self._params["tau_I"],
            "t_ref": self._params["t_ref"],
            "V_th": self._params["V_th_I"],
            "V_reset": self._params["V_r"],
            "I_e": I_xI
            if self._params["delta_I_xE"] == 0
            else I_xI * nest.random.uniform(1 - self._params["delta_I_xE"] / 2, 1 + self._params["delta_I_xE"] / 2),
            "tau_syn_ex": self._params["tau_syn_ex"],
            "tau_syn_in": self._params["tau_syn_in"],
            "V_m": self._params["V_m"]
            if not self._params["V_m"] == "rand"
            else self._params["V_th_I"] - 20 * nest.random.lognormal(0, 1),
        }

        # iaf_psc_exp allows stochasticity, if not used - don't supply the parameters and use
        # iaf_psc_exp as deterministic model
        if (self._params.get("delta") is not None) and (self._params.get("rho") is not None):
            E_neuron_params["delta"] = self._params["delta"]
            I_neuron_params["delta"] = self._params["delta"]
            E_neuron_params["rho"] = self._params["rho"]
            I_neuron_params["rho"] = self._params["rho"]

        # create the neuron populations
        pop_size_E = self._params["N_E"] // self._params["n_clusters"]
        pop_size_I = self._params["N_I"] // self._params["n_clusters"]
        if self._params.get("conn_seed") is not None: # control the random seed of each neuron's nest parameter when changing Q (Kim & Shin, 2026)
            print(f'Control randomness of V_m and connection')
            E_pops = [
                nest.Create(self._params["neuron_type"], n=1, params=E_neuron_params)
                for _ in range(self._params["N_E"])
            ]
            I_pops = [
                nest.Create(self._params["neuron_type"], n=1, params=I_neuron_params)
                for _ in range(self._params["N_I"])
            ]
            
            if self._params["n_clusters"] == 1 and self._params.get("multi_stim_neurons") is not None: # no cluster, but divide neurons for stimulation
                n_stims = len(np.unique(self._params["stim_inds_trial"]))
                n_clusters = n_stims
            else:
                n_clusters = self._params["n_clusters"]

            # cluster neurons
            temp_E_pops, temp_I_pops = [], []
            n_E_clust, n_I_clust = self._params["N_E"] // n_clusters, self._params["N_I"] // n_clusters
            for ind in range(n_clusters):
                all_units_E, all_units_I = E_pops[ind*n_E_clust], I_pops[ind*n_I_clust]
                for E_ind in np.arange(ind*n_E_clust + 1, (ind+1)*n_E_clust):
                    all_units_E += E_pops[E_ind]
                for I_ind in np.arange(ind*n_I_clust + 1, (ind+1)*n_I_clust):
                    all_units_I += I_pops[I_ind]
                temp_E_pops.append(all_units_E)
                temp_I_pops.append(all_units_I)
            E_pops, I_pops = temp_E_pops, temp_I_pops
        else:
            E_pops = [
                nest.Create(self._params["neuron_type"], n=pop_size_E, params=E_neuron_params)
                for _ in range(self._params["n_clusters"])
            ]
            I_pops = [
                nest.Create(self._params["neuron_type"], n=pop_size_I, params=I_neuron_params)
                for _ in range(self._params["n_clusters"])
            ]
        self._populations = [E_pops, I_pops]

    def connect(self):
        """Connect the excitatory and inhibitory populations with each other
        in the EI-clustered scheme

        Raises
        ------
        ValueError
            If the clustering method is not recognized
        """

        if "clustering" not in self._params or self._params["clustering"] == "weight":
            self.connect_weight()
        elif self._params["clustering"] == "probabilities":
            self.connect_probabilities()
        else:
            raise ValueError("Clustering method %s not implemented" % self._params["clustering"])

    def connect_probabilities(self):
        """Connect the clusters with a probability EI-cluster scheme

        Connects the excitatory and inhibitory populations with each other
        in the EI-clustered scheme by increasing the probabilities of the
        connections within the clusters and decreasing the probabilities of the
        connections between the clusters. The weights are calculated so that
        the total input to a neuron is balanced.
        """

        #  self._populations[0] -> Excitatory population
        #  self._populations[1] -> Inhibitory population

        N = self._params["N_E"] + self._params["N_I"]  # total units
        # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
        # if any of the js is nan or not given
        if self._params.get("js") is None or np.isnan(self._params.get("js")).any():
            js = helper.calculate_RBN_weights(self._params)
        js *= self._params["s"]

        if self._params["n_clusters"] > 1:
            pminus = (self._params["n_clusters"] - self._params["pplus"]) / float(self._params["n_clusters"] - 1)
        else:
            self._params["pplus"] = np.ones((2, 2))
            pminus = np.ones((2, 2))

        p_plus = self._params["pplus"] * self._params["baseline_conn_prob"]
        p_minus = pminus * self._params["baseline_conn_prob"]

        # Connection probabilities within clusters can exceed 1. In this case, we iteratively split
        # the connections in multiple synapse populations with probabilities < 1.
        iterations = np.ones((2, 2), dtype=int)
        # test if any of the probabilities is larger than 1
        if np.any(p_plus > 1):
            print("The probability of some connections is larger than 1.")
            print("Pre-splitting the connections in multiple synapse populations:")
            printoptions = np.get_printoptions()
            np.set_printoptions(precision=2, floatmode="fixed")
            print("p_plus:\n", p_plus)
            print("p_minus:\n", p_minus)
            for i in range(2):
                for j in range(2):
                    if p_plus[i, j] > 1:
                        iterations[i, j] = int(np.ceil(p_plus[i, j]))
                        p_plus[i, j] /= iterations[i, j]
            print("\nPost-splitting the connections in multiple synapse populations:")
            print("p_plus:\n", p_plus)
            print("Number of synapse populations:\n", iterations)
            np.set_printoptions(**printoptions)

        # define the synapses and connect the populations

        # Excitatory to excitatory neuron connections
        j_ee = js[0, 0] / np.sqrt(N)
        nest.CopyModel("static_synapse", "EE", {"weight": j_ee, "delay": self._params["delay"]})

        if self._params["fixed_indegree"]:
            K_EE_plus = int(p_plus[0, 0] * self._params["N_E"] / self._params["n_clusters"])
            print("K_EE+: ", K_EE_plus)
            K_EE_minus = int(p_minus[0, 0] * self._params["N_E"] / self._params["n_clusters"])
            print("K_EE-: ", K_EE_minus)
            conn_params_EE_plus = {
                "rule": "fixed_indegree",
                "indegree": K_EE_plus,
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_EE_minus = {
                "rule": "fixed_indegree",
                "indegree": K_EE_minus,
                "allow_autapses": False,
                "allow_multapses": True,
            }

        else:
            conn_params_EE_plus = {
                "rule": "pairwise_bernoulli",
                "p": p_plus[0, 0],
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_EE_minus = {
                "rule": "pairwise_bernoulli",
                "p": p_minus[0, 0],
                "allow_autapses": False,
                "allow_multapses": True,
            }
        for i, pre in enumerate(self._populations[0]):
            for j, post in enumerate(self._populations[0]):
                if i == j:
                    # same cluster
                    for n in range(iterations[0, 0]):
                        nest.Connect(pre, post, conn_params_EE_plus, "EE")
                else:
                    nest.Connect(pre, post, conn_params_EE_minus, "EE")

        # Inhibitory to excitatory neuron connections
        j_ei = js[0, 1] / np.sqrt(N)
        nest.CopyModel("static_synapse", "EI", {"weight": j_ei, "delay": self._params["delay"]})

        if self._params["fixed_indegree"]:
            K_EI_plus = int(p_plus[0, 1] * self._params["N_I"] / self._params["n_clusters"])
            print("K_EI+: ", K_EI_plus)
            K_EI_minus = int(p_minus[0, 1] * self._params["N_I"] / self._params["n_clusters"])
            print("K_EI-: ", K_EI_minus)
            conn_params_EI_plus = {
                "rule": "fixed_indegree",
                "indegree": K_EI_plus,
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_EI_minus = {
                "rule": "fixed_indegree",
                "indegree": K_EI_minus,
                "allow_autapses": False,
                "allow_multapses": True,
            }

        else:
            conn_params_EI_plus = {
                "rule": "pairwise_bernoulli",
                "p": p_plus[0, 1],
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_EI_minus = {
                "rule": "pairwise_bernoulli",
                "p": p_minus[0, 1],
                "allow_autapses": False,
                "allow_multapses": True,
            }
        for i, pre in enumerate(self._populations[1]):
            for j, post in enumerate(self._populations[0]):
                if i == j:
                    # same cluster
                    for n in range(iterations[0, 1]):
                        nest.Connect(pre, post, conn_params_EI_plus, "EI")
                else:
                    nest.Connect(pre, post, conn_params_EI_minus, "EI")

        # Excitatory to inhibitory neuron connections
        j_ie = js[1, 0] / np.sqrt(N)
        nest.CopyModel("static_synapse", "IE", {"weight": j_ie, "delay": self._params["delay"]})

        if self._params["fixed_indegree"]:
            K_IE_plus = int(p_plus[1, 0] * self._params["N_E"] / self._params["n_clusters"])
            print("K_IE+: ", K_IE_plus)
            K_IE_minus = int(p_minus[1, 0] * self._params["N_E"] / self._params["n_clusters"])
            print("K_IE-: ", K_IE_minus)
            conn_params_IE_plus = {
                "rule": "fixed_indegree",
                "indegree": K_IE_plus,
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_IE_minus = {
                "rule": "fixed_indegree",
                "indegree": K_IE_minus,
                "allow_autapses": False,
                "allow_multapses": True,
            }

        else:
            conn_params_IE_plus = {
                "rule": "pairwise_bernoulli",
                "p": p_plus[1, 0],
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_IE_minus = {
                "rule": "pairwise_bernoulli",
                "p": p_minus[1, 0],
                "allow_autapses": False,
                "allow_multapses": True,
            }
        for i, pre in enumerate(self._populations[0]):
            for j, post in enumerate(self._populations[1]):
                if i == j:
                    # same cluster
                    for n in range(iterations[1, 0]):
                        nest.Connect(pre, post, conn_params_IE_plus, "IE")
                else:
                    nest.Connect(pre, post, conn_params_IE_minus, "IE")

        # Inhibitory to inhibitory neuron connections
        j_ii = js[1, 1] / np.sqrt(N)
        nest.CopyModel("static_synapse", "II", {"weight": j_ii, "delay": self._params["delay"]})

        if self._params["fixed_indegree"]:
            K_II_plus = int(p_plus[1, 1] * self._params["N_I"] / self._params["n_clusters"])
            print("K_II+: ", K_II_plus)
            K_II_minus = int(p_minus[1, 1] * self._params["N_I"] / self._params["n_clusters"])
            print("K_II-: ", K_II_minus)
            conn_params_II_plus = {
                "rule": "fixed_indegree",
                "indegree": K_II_plus,
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_II_minus = {
                "rule": "fixed_indegree",
                "indegree": K_II_minus,
                "allow_autapses": False,
                "allow_multapses": True,
            }

        else:
            conn_params_II_plus = {
                "rule": "pairwise_bernoulli",
                "p": p_plus[1, 1],
                "allow_autapses": False,
                "allow_multapses": True,
            }
            conn_params_II_minus = {
                "rule": "pairwise_bernoulli",
                "p": p_minus[1, 1],
                "allow_autapses": False,
                "allow_multapses": True,
            }
        for i, pre in enumerate(self._populations[1]):
            for j, post in enumerate(self._populations[1]):
                if i == j:
                    # same cluster
                    for n in range(iterations[1, 1]):
                        nest.Connect(pre, post, conn_params_II_plus, "II")
                else:
                    nest.Connect(pre, post, conn_params_II_minus, "II")

    def connect_weight(self):
        """Connect the clusters with a weight EI-cluster scheme

        Connects the excitatory and inhibitory populations with
        each other in the EI-clustered scheme by increasing the weights
        of the connections within the clusters and decreasing the weights
        of the connections between the clusters. The weights are calculated
        so that the total input to a neuron is balanced.
        """

        #  self._populations[0] -> Excitatory population
        #  self._populations[1] -> Inhibitory population

        N = self._params["N_E"] + self._params["N_I"]  # total units

        # if js are not given compute them so that sqrt(K) spikes equal v_thr-E_L and rows are balanced
        # if any of the js is nan or not given
        if self._params.get("js") is None or np.isnan(self._params.get("js")).any():
            js = helper.calculate_RBN_weights(self._params)
        js *= self._params["s"]

        # jminus is calculated so that row sums remain constant
        if self._params["n_clusters"] > 1:
            if self._params["rep"] is not None and not isinstance(self._params["rep"], str): # revised from original version of Rostami et al., 2024
                jminus = (self._params["n_clusters"] - self._params["jplus"]) / float(self._params["n_clusters"] - 1)
            elif isinstance(self._params["rep"], str):
                if self._params["rep"] == 'm4': # Kim & Shin, 2026
                    print(f'Method 4')
                    # method 2: match the difference of intra-cluster weight vs. inter-cluster weight
                    jep_cri = 3.2; n_clusters_cri = 6; jip_cri = 1.0 + (jep_cri - 1) * self._params["rj"]
                    kE = jep_cri - (n_clusters_cri - jep_cri) / (n_clusters_cri - 1); kI = jip_cri - (n_clusters_cri - jip_cri) / (n_clusters_cri - 1)
                    
                    jep = kE + 1 - kE/self._params["n_clusters"]
                    jip = kI + 1 - kI/self._params["n_clusters"]
                    self._params["jplus"] = np.array([[jep, jip], [jip, jip]])
                    jminus = (self._params["n_clusters"] - self._params["jplus"]) / float(self._params["n_clusters"] - 1)

        else:
            self._params["jplus"] = np.ones((2, 2))
            jminus = np.ones((2, 2))

        # define the synapses and connect the populations

        # Excitatory to excitatory neuron connections
        j_ee = js[0, 0] / np.sqrt(N)
        self._params["j_ee"] = j_ee # revised from original version of Rostami et al., 2024
        nest.CopyModel(
            "static_synapse",
            "EE_plus",
            {
                "weight": self._params["jplus"][0, 0] * j_ee,
                "delay": self._params["delay"],
            },
        )
        nest.CopyModel(
            "static_synapse",
            "EE_minus",
            {"weight": jminus[0, 0] * j_ee, "delay": self._params["delay"]},
        )
        if self._params["fixed_indegree"]:
            K_EE = int(self._params["baseline_conn_prob"][0, 0] * self._params["N_E"] / self._params["n_clusters"])
            print("K_EE: ", K_EE)
            conn_params_EE = {
                "rule": "fixed_indegree",
                "indegree": K_EE,
                "allow_autapses": False,
                "allow_multapses": False,
            }

        else:
            if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                conn_params_EE = "one_to_one"
                
                # connection matrix (row: post, col: pre)
                rng_conn = np.random.default_rng(self._params["conn_seed"])
                n_offdiag = self._params["N_E"] * self._params["N_E"] - self._params["N_E"]
                adj_mat_offdiag = rng_conn.random(n_offdiag) < self._params["baseline_conn_prob"][0, 0]
                adj_mat = np.zeros((self._params["N_E"], self._params["N_E"]))
                adj_mat[~np.eye(self._params["N_E"], dtype=bool)] = adj_mat_offdiag.astype(int).copy()
            else:
                conn_params_EE = {
                    "rule": "pairwise_bernoulli",
                    "p": self._params["baseline_conn_prob"][0, 0],
                    "allow_autapses": False,
                    "allow_multapses": False,
                }
        for i, pre in enumerate(self._populations[0]):
            for j, post in enumerate(self._populations[0]):
                if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                    if self._params["n_clusters"] == 1 and self._params.get("multi_stim_neurons") is not None: # no cluster, but divide neurons for stimulation
                        n_stims = len(np.unique(self._params["stim_inds_trial"]))
                        n_clusters = n_stims
                    else:
                        n_clusters = self._params["n_clusters"]

                    n_E_clust = self._params["N_E"] // n_clusters
                    adj_mat_clust = adj_mat[:, i*n_E_clust:(i+1)*n_E_clust][j*n_E_clust:(j+1)*n_E_clust].copy()
                    post_ids, pre_ids = np.nonzero(adj_mat_clust)
                    post_ids += j*n_E_clust + 1 # nest node id starts from 1
                    pre_ids += i*n_E_clust + 1 # nest node id starts from 1
                    if len(pre_ids) > 0:
                        if i == j:
                            # same cluster
                            nest.Connect(pre_ids, post_ids, conn_params_EE, "EE_plus")
                        else:
                            nest.Connect(pre_ids, post_ids, conn_params_EE, "EE_minus")
                else:
                    if i == j:
                        # same cluster
                        nest.Connect(pre, post, conn_params_EE, "EE_plus")
                    else:
                        nest.Connect(pre, post, conn_params_EE, "EE_minus")

        # Inhibitory to excitatory neuron connections
        j_ei = js[0, 1] / np.sqrt(N)
        self._params["j_ei"] = j_ei # revised from original version of Rostami et al., 2024
        nest.CopyModel(
            "static_synapse",
            "EI_plus",
            {
                "weight": j_ei * self._params["jplus"][0, 1],
                "delay": self._params["delay"],
            },
        )
        nest.CopyModel(
            "static_synapse",
            "EI_minus",
            {"weight": j_ei * jminus[0, 1], "delay": self._params["delay"]},
        )
        if self._params["fixed_indegree"]:
            K_EI = int(self._params["baseline_conn_prob"][0, 1] * self._params["N_I"] / self._params["n_clusters"])
            print("K_EI: ", K_EI)
            conn_params_EI = {
                "rule": "fixed_indegree",
                "indegree": K_EI,
                "allow_autapses": False,
                "allow_multapses": False,
            }
        else:
            if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                conn_params_EI = "one_to_one"
                
                # connection matrix (row: post, col: pre)
                rng_conn = np.random.default_rng(self._params["conn_seed"])
                n_pair = self._params["N_E"] * self._params["N_I"]
                adj_mat = rng_conn.random(n_pair) < self._params["baseline_conn_prob"][0, 1]
                adj_mat = adj_mat.reshape(self._params["N_E"], self._params["N_I"]).astype(int)
            else:            
                conn_params_EI = {
                    "rule": "pairwise_bernoulli",
                    "p": self._params["baseline_conn_prob"][0, 1],
                    "allow_autapses": False,
                    "allow_multapses": False,
                }
        for i, pre in enumerate(self._populations[1]):
            for j, post in enumerate(self._populations[0]):
                if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                    if self._params["n_clusters"] == 1 and self._params.get("multi_stim_neurons") is not None: # no cluster, but divide neurons for stimulation
                        n_stims = len(np.unique(self._params["stim_inds_trial"]))
                        n_clusters = n_stims
                    else:
                        n_clusters = self._params["n_clusters"]

                    n_E_clust, n_I_clust = self._params["N_E"] // n_clusters, self._params["N_I"] // n_clusters
                    adj_mat_clust = adj_mat[:, i*n_I_clust:(i+1)*n_I_clust][j*n_E_clust:(j+1)*n_E_clust].copy()
                    post_ids, pre_ids = np.nonzero(adj_mat_clust)
                    post_ids += j*n_E_clust + 1 # nest node id starts from 1
                    pre_ids += self._params["N_E"] + i*n_I_clust + 1 # nest node id starts from 1
                    if len(pre_ids) > 0:
                        if i == j:
                            # same cluster
                            nest.Connect(pre_ids, post_ids, conn_params_EI, "EI_plus")
                        else:
                            nest.Connect(pre_ids, post_ids, conn_params_EI, "EI_minus")
                else:
                    if i == j:
                        # same cluster
                        nest.Connect(pre, post, conn_params_EI, "EI_plus")
                    else:
                        nest.Connect(pre, post, conn_params_EI, "EI_minus")

        # Excitatory to inhibitory neuron connections
        j_ie = js[1, 0] / np.sqrt(N)
        self._params["j_ie"] = j_ie # revised from original version of Rostami et al., 2024
        nest.CopyModel(
            "static_synapse",
            "IE_plus",
            {
                "weight": j_ie * self._params["jplus"][1, 0],
                "delay": self._params["delay"],
            },
        )
        nest.CopyModel(
            "static_synapse",
            "IE_minus",
            {"weight": j_ie * jminus[1, 0], "delay": self._params["delay"]},
        )

        if self._params["fixed_indegree"]:
            K_IE = int(self._params["baseline_conn_prob"][1, 0] * self._params["N_E"] / self._params["n_clusters"])
            print("K_IE: ", K_IE)
            conn_params_IE = {
                "rule": "fixed_indegree",
                "indegree": K_IE,
                "allow_autapses": False,
                "allow_multapses": False,
            }
        else:
            if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                conn_params_IE = "one_to_one"
                
                # connection matrix (row: post, col: pre)
                rng_conn = np.random.default_rng(self._params["conn_seed"])
                n_pair = self._params["N_I"] * self._params["N_E"]
                adj_mat = rng_conn.random(n_pair) < self._params["baseline_conn_prob"][1, 0]
                adj_mat = adj_mat.reshape(self._params["N_I"], self._params["N_E"]).astype(int)
            else:             
                conn_params_IE = {
                    "rule": "pairwise_bernoulli",
                    "p": self._params["baseline_conn_prob"][1, 0],
                    "allow_autapses": False,
                    "allow_multapses": False,
                }
        for i, pre in enumerate(self._populations[0]):
            for j, post in enumerate(self._populations[1]):
                if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                    if self._params["n_clusters"] == 1 and self._params.get("multi_stim_neurons") is not None: # no cluster, but divide neurons for stimulation
                        n_stims = len(np.unique(self._params["stim_inds_trial"]))
                        n_clusters = n_stims
                    else:
                        n_clusters = self._params["n_clusters"]

                    n_E_clust, n_I_clust = self._params["N_E"] // n_clusters, self._params["N_I"] // n_clusters
                    adj_mat_clust = adj_mat[:, i*n_E_clust:(i+1)*n_E_clust][j*n_I_clust:(j+1)*n_I_clust].copy()
                    post_ids, pre_ids = np.nonzero(adj_mat_clust)
                    post_ids += self._params["N_E"] + j*n_I_clust + 1 # nest node id starts from 1
                    pre_ids += i*n_E_clust + 1 # nest node id starts from 1
                    if len(pre_ids) > 0:
                        if i == j:
                            # same cluster
                            nest.Connect(pre_ids, post_ids, conn_params_IE, "IE_plus")
                        else:
                            nest.Connect(pre_ids, post_ids, conn_params_IE, "IE_minus")
                else:
                    if i == j:
                        # same cluster
                        nest.Connect(pre, post, conn_params_IE, "IE_plus")
                    else:
                        nest.Connect(pre, post, conn_params_IE, "IE_minus")

        # Inhibitory to inhibitory neuron connections
        j_ii = js[1, 1] / np.sqrt(N)
        self._params["j_ii"] = j_ii # revised from original version of Rostami et al., 2024
        nest.CopyModel(
            "static_synapse",
            "II_plus",
            {
                "weight": j_ii * self._params["jplus"][1, 1],
                "delay": self._params["delay"],
            },
        )
        nest.CopyModel(
            "static_synapse",
            "II_minus",
            {"weight": j_ii * jminus[1, 1], "delay": self._params["delay"]},
        )
        if self._params["fixed_indegree"]:
            K_II = int(self._params["baseline_conn_prob"][1, 1] * self._params["N_I"] / self._params["n_clusters"])
            print("K_II: ", K_II)
            conn_params_II = {
                "rule": "fixed_indegree",
                "indegree": K_II,
                "allow_autapses": False,
                "allow_multapses": False,
            }
        else:
            if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                conn_params_II = "one_to_one"
                
                # connection matrix (row: post, col: pre)
                rng_conn = np.random.default_rng(self._params["conn_seed"])
                n_offdiag = self._params["N_I"] * self._params["N_I"] - self._params["N_I"]
                adj_mat_offdiag = rng_conn.random(n_offdiag) < self._params["baseline_conn_prob"][1, 1]
                adj_mat = np.zeros((self._params["N_I"], self._params["N_I"]))
                adj_mat[~np.eye(self._params["N_I"], dtype=bool)] = adj_mat_offdiag.astype(int).copy()
            else:            
                conn_params_II = {
                    "rule": "pairwise_bernoulli",
                    "p": self._params["baseline_conn_prob"][1, 1],
                    "allow_autapses": False,
                    "allow_multapses": False,
                }
        for i, pre in enumerate(self._populations[1]):
            for j, post in enumerate(self._populations[1]):
                if self._params.get("conn_seed") is not None: # Kim & Shin, 2026
                    if self._params["n_clusters"] == 1 and self._params.get("multi_stim_neurons") is not None: # no cluster, but divide neurons for stimulation
                        n_stims = len(np.unique(self._params["stim_inds_trial"]))
                        n_clusters = n_stims
                    else:
                        n_clusters = self._params["n_clusters"]

                    n_I_clust = self._params["N_I"] // n_clusters
                    adj_mat_clust = adj_mat[:, i*n_I_clust:(i+1)*n_I_clust][j*n_I_clust:(j+1)*n_I_clust].copy()
                    post_ids, pre_ids = np.nonzero(adj_mat_clust)
                    post_ids += self._params["N_E"] + j*n_I_clust + 1 # nest node id starts from 1
                    pre_ids += self._params["N_E"] + i*n_I_clust + 1 # nest node id starts from 1
                    if len(pre_ids) > 0:
                        if i == j:                
                            # same cluster
                            nest.Connect(pre_ids, post_ids, conn_params_II, "II_plus")
                        else:
                            nest.Connect(pre_ids, post_ids, conn_params_II, "II_minus")
                else:
                    if i == j:
                        # same cluster
                        nest.Connect(pre, post, conn_params_II, "II_plus")
                    else:
                        nest.Connect(pre, post, conn_params_II, "II_minus")

    def create_stimulation(self):
        """Create a current source and connect it to clusters."""

        if self._params.get('gen_name') is None:
            gen_name = "step_current_generator"

        if "stim_clusters" in self._params: # not used in Kim & Shin, 2026
            stim_amp = self._params["stim_amp"]  # amplitude of the stimulation current in pA
            stim_starts = self._params["stim_starts"]  # list of stimulation start times
            stim_ends = self._params["stim_ends"]  # list of stimulation end times
            amplitude_values = []
            amplitude_times = []
            for start, end in zip(stim_starts, stim_ends):
                amplitude_times.append(start + self._params["warmup"])
                amplitude_values.append(stim_amp)
                amplitude_times.append(end + self._params["warmup"])
                amplitude_values.append(0.0)
            self._currentsources = [nest.Create(gen_name)]
            for stim_cluster in self._params["stim_clusters"]:
                nest.Connect(self._currentsources[0], self._populations[0][stim_cluster])
            nest.SetStatus(
                self._currentsources[0],
                {
                    "amplitude_times": amplitude_times,
                    "amplitude_values": amplitude_values,
                    "allow_offgrid_times": True
                },
            )
        elif "multi_stim_clusters" in self._params:
            print('stimulating multi stim ...')
            for stim_ind, (stim_clusters, amplitudes, times) in enumerate(zip(self._params['multi_stim_clusters'],
                                                                            self._params['multi_stim_amps'],
                                                                            self._params['multi_stim_times'])):
                if self._params.get('gen_name') is None: # revised from original version of Rostami et al., 2024
                    self._currentsources.append(nest.Create(gen_name))
                    nest.SetStatus(self._currentsources[-1],
                                {'amplitude_times': list(np.array(times) + self._params["warmup"]),
                                    'amplitude_values': amplitudes,
                                    'allow_offgrid_times': True})
                    stim_units = []
                    for stim_cluster in stim_clusters:
                        nest.Connect(self._currentsources[-1], 
                                    self._populations[0][stim_cluster])

        elif "multi_stim_neurons" in self._params: # Q = 1 (revised from original version of Rostami et al., 2024)
            print('stimulating multi neu ...')
            for ind, (stim_neurons, amplitudes, times) in enumerate(zip(self._params['multi_stim_neurons'],
                                                                        self._params['multi_stim_amps'],
                                                                        self._params['multi_stim_times'])):
                if self._params.get('gen_name') is None:
                    self._currentsources.append(nest.Create(gen_name))
                    nest.SetStatus(self._currentsources[-1],
                                {'amplitude_times': list(np.array(times) + self._params["warmup"]),
                                    'amplitude_values': amplitudes,
                                    'allow_offgrid_times': True})
                    stim_units = []
                    if self._params["n_clusters"] == 1 and self._params.get("multi_stim_neurons") is not None: # no cluster, but divide neurons for stimulation
                        nest.Connect(self._currentsources[-1], 
                                    self._populations[0][ind]) # ind is cluster index!
                    else:
                        for stim_neuron in stim_neurons:
                            neu_ind = np.where(self._params['list_stim_neurons_ind'][:, 0] == stim_neuron)[0][0] # index among the stimulated neurons
                            _, cluster_ind, intracluster_ind = self._params['list_stim_neurons_ind'][neu_ind]
                            nest.Connect(self._currentsources[-1], 
                                        self._populations[0][cluster_ind][intracluster_ind])

    def create_recording_devices(self):
        """Creates a spike recorder

        Create and connect a spike recorder to all neuron populations
        in self._populations.
        """
        self._recording_devices = [nest.Create("spike_recorder")]
        self._recording_devices[0].record_to = "memory"

        all_units = self._populations[0][0]
        for E_pop in self._populations[0][1:]:
            all_units += E_pop
        for I_pop in self._populations[1]:
            all_units += I_pop
        nest.Connect(all_units, self._recording_devices[0], "all_to_all")  # Spikerecorder

    def set_model_build_pipeline(self, pipeline):
        """Set _model_build_pipeline

        Parameters
        ----------
        pipeline: list
            ordered list of functions executed to build the network model
        """
        self._model_build_pipeline = pipeline

    def setup_network(self):
        """Setup network in NEST

        Initializes NEST and creates
        the network in NEST, ready to be simulated.
        Functions saved in _model_build_pipeline are executed.
        """
        for func in self._model_build_pipeline:
            func()

    def simulate(self):
        """Simulates network for a period of warmup+simtime"""
        nest.Simulate(self._params["warmup"] + self._params["simtime"])

    def get_recordings(self):
        """Extract spikes from Spikerecorder

        Extract spikes form the Spikerecorder connected
        to all populations created in create_populations.
        Cuts the warmup period away and sets time relative to end of warmup.
        Ids 1:N_E correspond to excitatory neurons,
        N_E+1:N_E+N_I correspond to inhibitory neurons.

        Returns
        -------
        spiketimes: ndarray
            2D array [2xN_Spikes]
            of spiketimes with spiketimes in row 0 and neuron IDs in row 1.
        """
        events = nest.GetStatus(self._recording_devices[0], "events")[0]
        # convert them to the format accepted by spiketools
        spiketimes = np.append(events["times"][None, :], events["senders"][None, :], axis=0)
        spiketimes[1] -= 1
        # remove the pre warmup spikes
        spiketimes = spiketimes[:, spiketimes[0] >= self._params["warmup"]]
        spiketimes[0] -= self._params["warmup"]
        return spiketimes

    def get_parameter(self):
        """Get all parameters used to create the network.
        Returns
        -------
        dict
            Dictionary with all parameters of the network and the simulation.
        """
        return self._params

    def create_and_simulate(self):
        """Create and simulate the EI-clustered network.

        Returns
        -------
        spiketimes: ndarray
            2D array [2xN_Spikes]
            of spiketimes with spiketimes in row 0 and neuron IDs in row 1.
        """
        self.setup_network()
        self.simulate()
        return self.get_recordings()

    def get_firing_rates(self, spiketimes=None):
        """Calculates the average firing rates of
        all excitatory and inhibitory neurons.

        Calculates the firing rates of all excitatory neurons
        and the firing rates of all inhibitory neurons
        created by self.create_populations.
        If spiketimes are not supplied, they get extracted.

        Parameters
        ----------
        spiketimes: ndarray
            2D array [2xN_Spikes] of spiketimes
            with spiketimes in row 0 and neuron IDs in row 1.

        Returns
        -------
        tuple[float, float]
            average firing rates of excitatory (0)
            and inhibitory (1) neurons (spikes/s)
        """
        if spiketimes is None:
            spiketimes = self.get_recordings()
        e_count = spiketimes[:, spiketimes[1] < self._params["N_E"]].shape[1]
        i_count = spiketimes[:, spiketimes[1] >= self._params["N_E"]].shape[1]
        e_rate = e_count / float(self._params["N_E"]) / float(self._params["simtime"]) * 1000.0
        i_rate = i_count / float(self._params["N_I"]) / float(self._params["simtime"]) * 1000.0
        return e_rate, i_rate

    def set_I_x(self, I_XE, I_XI):
        """Set DC currents for excitatory and inhibitory neurons
        Adds DC currents for the excitatory and inhibitory neurons.
        The DC currents are added to the currents already
        present in the populations.

        Parameters
        ----------
        I_XE: float
            extra DC current for excitatory neurons [pA]
        I_XI: float
            extra DC current for inhibitory neurons [pA]
        """
        for E_pop in self._populations[0]:
            I_e_loc = E_pop.get("I_e")
            E_pop.set({"I_e": I_e_loc + I_XE})
        for I_pop in self._populations[1]:
            I_e_loc = I_pop.get("I_e")
            I_pop.set({"I_e": I_e_loc + I_XI})

    def get_simulation(self, PathSpikes=None):
        """Create network, simulate and return results

        Creates the EI-clustered network and simulates it with
        the parameters supplied in the object creation.
        Returns a dictionary with firing rates,
        timing information (dict) and parameters (dict).
        If PathSpikes is supplied the spikes get saved to a pickle file.

        Parameters
        ----------
        PathSpikes: str (optional)
            Path of file for spiketimes, if None, no file is saved

        Returns
        -------
        dict
         Dictionary with firing rates,
         spiketimes (ndarray) and parameters (dict)
        """

        self.setup_network()
        self.simulate()
        spiketimes = self.get_recordings()
        e_rate, i_rate = self.get_firing_rates(spiketimes)
        
        # weight matrix
        conn_vals = nest.GetConnections().get(['source', 'target', 'weight'])
        N_E, N_I = self._params['N_E'], self._params['N_I']
        N = N_E + N_I

        # total_nodes = nest.GetKernelStatus("network_size")
        # print(f"Total nodes in network: {total_nodes}")
        # print(nest.GetStatus(nest.NodeCollection(np.arange(N+1, total_nodes+1)), 'model')) # last nodes are not neurons
        
        adjmat = np.zeros((N, N))
        source_inds = np.array(conn_vals['source']) - 1 # node ID starts from 1, so subtract 1 to make python index
        target_inds = np.array(conn_vals['target']) - 1
        weights = np.array(conn_vals['weight'])
        bool_neu = (source_inds < N) & (target_inds < N) # nodes with index > N are not neurons; they are Q current sources and one recording device
        source_inds = source_inds[bool_neu]
        target_inds = target_inds[bool_neu]
        weights = weights[bool_neu]
        adjmat[source_inds, target_inds] = weights
        
        if PathSpikes is not None:
            with gzip.open(PathSpikes, "wb") as outfile:
                pickle.dump(spiketimes, outfile)
        return {
            "e_rate": e_rate,
            "i_rate": i_rate,
            "_params": self.get_parameter(),
            "spiketimes": spiketimes,
            "adjmat": adjmat
        }