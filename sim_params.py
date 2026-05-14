n_stims = 6
n_trials = 50
dur_stim = 250 # ms
simtime = float(n_stims*n_trials*dur_stim)
sim_dict = {
    # The full simulation time is the sum of a presimulation time and the main
    # simulation time.
    # presimulation time (in ms)
    "warmup": 1000.0,
    # simulation time (in ms)
    "simtime": simtime,
    # resolution of the simulation (in ms)
    "dt": 0.1,
    "randseed": 55,
    # Number of virtual processes
    "n_vp": 4
}