### 1. File description

This repository contains the source code for the manuscript "Neural variability structure in primary visual cortex is optimal for robust representation of visual similarity" (Kim and Shin, 2025). We have 3 types of files:

- Jupyter Notebook files (`.ipynb`)  
- Python scripts (`.py`)  
- MATLAB scripts (`.m`)  

**To generate the figures, you only need the Jupyter Notebook files.**

### 2. How to generate figures

In each Jupyter Notebook, the first markdown cell and the cell immediately before each figure-generating cell include the figure citation. In summary:

- **box_counting_analysis.ipynb** → Figure S5C  
- **decode_noise_corr.ipynb** → Figure 2B, S4, S6A  
- **dimensionality_tangent.ipynb** → Figure 1D, 2D  
- **fit_slope_histogram.ipynb** → Figure 1B, 1C, S1, S2, S3, S5A  
- **fit_slope_openscope.ipynb** → No figures (used only for fitting log-mean vs. log-variance slope to OpenScope data)  
- **local_global_alignment.ipynb** → Figure 3E, 3F  
- **ortho_variance.ipynb** → Figure 3A–D  
- **RSA_overlap_SCC.ipynb** → Figure 2A, 2C, 4, S4C, S6B, S6C  
- **visualize_manifold.ipynb** → Figure 1C, S5B, S5D  

### 3. Details on the workflow

1. We downloaded the Allen Brain Observatory Visual Coding Neuropixels data (NWB files), and ran the two MATLAB scripts to extract single-unit spike counts and saved them as `.mat` files.  
   For instructions on downloading NWB files, see the AllenSDK example:  
   https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html#Using-the-AllenSDK-to-retrieve-data

2. We analyzed the spike counts in the `.mat` files using Jupyter notebooks and Python scripts, and saved the resulting variables into pickle files.

- The notebooks and Python scripts have been verified to run in a Python 3.11.10 virtual environment after installing the packages listed in `requirements.txt` (in the `code` folder).  
- The MATLAB scripts have been verified to run in MATLAB R2024b.
