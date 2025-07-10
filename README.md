### 1. File description

This repository contains the source code for the manuscript "Neural variability structure in primary visual cortex is optimal for robust representation of visual similarity" (Kim and Shin, 2025). We have 3 types of files:

- Jupyter Notebook files (`.ipynb`)  
- Python scripts (`.py`)  
- MATLAB scripts (`.m`)  

### 2. How to run the code
- The Jupyter Notebooks and Python scripts have been verified to run using Visual Studio Code (Windows 11) in a Python 3.11.10 virtual environment after installing the packages listed in `requirements.txt` using pip 24.2. (Save `requirements.txt` in the same directory as the Jupyter Notebooks.)
- The MATLAB scripts have been verified to run in MATLAB R2024b.

**To generate the figures, you only need the Jupyter Notebooks, and pickle files which can be downloaded from https://figshare.com/s/64957e60fe943cb732a7. Those pickle files contain all the variables needed to run the Jupyter Notebooks, so save them in the same directory as the Jupyter Notebooks and run each Jupyter Notebook from the first cell.**

First, install the packages listed in `requirements.txt` in your Python 3.11.10 virtual environment.
```bash
python -m pip install pip==24.2
python -m pip install -r requirements.txt
```

Second, run each Jupyter Notebook file using Visual Studio Code. You should run the cells in order, from the first cell. You can run specific Jupyter Notebooks for corresponding figures; in each Jupyter Notebook, the first markdown cell and the cell immediately before each figure-generating cell include the figure citation. In summary:

- **box_counting_analysis.ipynb** → Figure S5C  
- **decode_noise_corr.ipynb** → Figure 2B, S4, S6A  
- **dimensionality_tangent.ipynb** → Figure 1D, 2D  
- **fit_slope_histogram.ipynb** → Figure 1B, 1C, S1, S2, S3, S5A  
- **fit_slope_openscope.ipynb** → No figures (used only for fitting log-mean vs. log-variance slope to OpenScope data)  
- **local_global_alignment.ipynb** → Figure 3E, 3F  
- **ortho_variance.ipynb** → Figure 3A–D  
- **RSA_overlap_SCC.ipynb** → Figure 2A, 2C, 4, S4C, S6B, S6C  
- **visualize_manifold.ipynb** → Figure 1C, S5B, S5D  

### 3. How we collected and analyzed data

1. We downloaded the Allen Brain Observatory Visual Coding Neuropixels data (NWB files), and ran the two MATLAB scripts to extract single-unit spike counts and saved them as `.mat` files.  
   For instructions on downloading NWB files, see the AllenSDK example:  
   https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html#Using-the-AllenSDK-to-retrieve-data

2. We analyzed the spike counts in the `.mat` files using Jupyter Notebooks and Python scripts, and saved the resulting variables into pickle files.