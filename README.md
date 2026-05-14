## 1. File description

This repository contains the source code for the manuscript "Intrinsic geometry of trial-to-trial variability in primary visual cortex is optimal for robust representation of visual similarity" (Kim and Shin, 2026). We have 3 types of files:

- Jupyter Notebook files (`.ipynb`)  
- Python scripts (`.py`)  
- MATLAB scripts (`.m`)  

## 2. How to run the code
### 2-1. Minimal information to reproduce figures
**To generate the figures, you only need the Jupyter Notebooks, and pickle.gz files which can be downloaded from https://figshare.com/s/64957e60fe943cb732a7. Those pickle.gz files contain all the variables needed to run the Jupyter Notebooks, so save them in the same directory as the Jupyter Notebooks and run each Jupyter Notebook.**

First, install the packages listed in `requirements.txt` in a Python 3.11.10 virtual environment (Save `requirements.txt` in the same directory as the Jupyter Notebooks).
```cmd
python -m pip install pip==24.2
python -m pip install -r requirements.txt
```
Second, run each Jupyter Notebook file using Visual Studio Code. You should first run the helper codes at the top of the Notebook. Then, you can independently run specific cells for figures you want (If there are multiple cells for a figure panel, you should run them in order). In each Notebook, figure citations are at the top and immediately before each figure-generating cell. In summary:

- **decode_noise_corr.ipynb** → Figure 3b, Supplementary Figure 6a, 10a, b, e, 12b, 15b
- **dimensionality.ipynb** → Figure 2c, Supplementary Figure 10c, d, 12a, 15a
- **discrete_spkcnt.ipynb** → Figure 2b, 5, Supplementary Figure 1a-c, e, f, 11
- **divide_neu_sample_trials.ipynb** → Figure 6, Supplementary Figure 7, 9
- **gratings_spt_diffwin.ipynb** → Figure 1h, i, Supplementary Figure 2a, b, 8, 14
- **local_global_alignment.ipynb** → Supplementary Figure 5e, f  
- **monkey_analysis.ipynb** → Figure 1g, Supplementary Figure 15
- **ortho_variance.ipynb** → Supplementary Figure 5b-d  
- **RSA_overlap_SCC.ipynb** → Figure 3a, c, 4, Supplementary Figure 6b, 10f-h, 12c, d, 13, 15c, d
- **slope_2p_cal.ipynb** → Supplementary Figure 3
- **slope_change_detection.ipynb** → Figure 1k-n
- **slope_nat_scenes_movie.ipynb** → Figure 1a-f, j, 2d, 5a, Supplementary Figure 1d, 2c-f, 4a
- **spiking_network_analysis.ipynb** → Figure 7, Supplementary Figure 16
- **visualize_manifold.ipynb** → Figure 2e, Supplementary Figure 4b

### 2-2. Environment settings
- The Jupyter Notebooks and Python scripts, except for those inside the folder 'spiking network simulation', have been verified to run in a Python 3.11.10 virtual environment using Visual Studio Code (Windows 11).
- The MATLAB scripts have been verified to run in MATLAB R2024b.
- The Python scripts inside the 'spiking network simulation' folder are for simulation of spiking networks in Linux (Rostami et al., Nat. Commun., 2024). They have been verified to run in a Python 3.11.13 virtual environment (WSL2 Ubuntu 24.04). They were modified from the publicly available code based on the NEST Simulator package (https://nest-simulator.readthedocs.io/en/stable/auto_examples/EI_clustered_network/index.html). To simulate the spiking network, follow the instructions below:

First, in Windows PowerShell, install Ubuntu 24.04. You can specify the distro name, for example, 'Ubuntu-24.04-test'.
```powershell
wsl --install -d Ubuntu-24.04 --name Ubuntu-24.04-test
```
Then, in the Ubuntu terminal, install Miniforge to use Mamba (https://github.com/conda-forge/miniforge). You can specify where Miniforge is installed, for example, $HOME/miniforge3.
```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
mamba shell init --shell bash --root-prefix="$HOME/miniforge3"
exec bash
```
Finally, create a virtual environment using 'ei_network_explicit.txt' which contains the python packages including NEST. You can specify the environment name, for example, 'ei_network'.
```bash
EXPLICIT="/mnt/your_path/ei_network_explicit.txt"
mamba create -n ei_network --file "$EXPLICIT"
```
You can run simulation in the created environment. You should be in the directory containing the simulation scripts ('spiking network simulation' folder).
```bash
cd "/mnt/your_path/spiking network simulation"
mamba activate ei_network
python run_simulation.py
```

## 3. How we collected and analyzed data
### 3-1. Extracellular electrophysiology datasets
We downloaded two extracellular electrophysiology datasets using NWB files: Allen Brain Observatory Visual Coding Neuropixels and Visual Behavior Neuropixels. We then extracted single-unit spike counts in MATLAB and saved them as `.mat` files.
   For instructions on downloading NWB files, see the AllenSDK example:
   https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html#Using-the-AllenSDK-to-retrieve-data
   https://allensdk.readthedocs.io/en/latest/_static/examples/nb/visual_behavior_neuropixels_data_access.html

We also downloaded one publicly available dataset of monkey V1 single-unit spike counts (Cadena et al., PLoS Comp. Biol., 2024) from https://figshare.com/articles/dataset/Monkey_V1_responses_to_natural_images_from_Cadena_et_al_2023/23056805?backTo=%2Fcollections%2FMonkey_V1_and_V4_single-cell_responses_to_natural_images_ephys_Data_from_Cadena_et_al_2024_%2F6658331&file=40805201.

We analyzed the spike counts in Jupyter Notebooks and Python scripts, and saved the resulting variables into pickle.gz files.

### 3-2. Two-photon calcium imaging dataset
We downloaded one two-photon calcium imaging dataset: Allen Brain Observatory Visual Coding Optical Physiology. We extracted and analyzed single-unit dF/F and deconvolved event counts in a Jupyter Notebook based on AllenSDK (https://allensdk.readthedocs.io/en/latest/brain_observatory.html).
