# Poisson Generalized Additive Model (PGAM)

This repository provides tools for estimating neuronal tuning curves from spike count data using a Poisson Generalized Additive Model (PGAM). The PGAM tool was originally developed by Eduardo Bolzani in Christina Savin's group, and the original codebase can be found at [Savin-Lab-Code/PGAM](https://github.com/Savin-Lab-Code/PGAM). This repository adapts that tool for application to data collected in the Smear Lab. Tuning curves are constructed with B-spline bases whose knots and order are specified in `make_config`, and results can be plotted to assess significance.

## Setup
1. Install [R](https://www.r-project.org/) and [Anaconda](https://www.anaconda.com/). Add R to your system path and define an `R_HOME` environment variable pointing to the R installation.
2. Create the conda environment and install packages:
   ```bash
   conda create -n pgam python=3.9 -y
   conda activate pgam

   conda install -y -c conda-forge numpy==1.20.3 numba==0.55.2 scipy==1.5.3 scikit-learn==1.1.2 pandas==1.3.3 dill==0.3.3 statsmodels==0.12.2
   conda install matplotlib seaborn pyyaml h5py ipykernel

   pip install rpy2==3.4.4 opt_einsum==3.3.0
   ```
3. Install the required R utilities:
   ```python
   from rpy2.robjects.packages import importr
   utils = importr('utils')
   utils.chooseCRANmirror(ind=1)  # select a CRAN mirror
   utils.install_packages('survey')
   ```
4. Test the environment:
   ```python
   from GAM_library import *
   ```

## Usage

The main entry point for fitting models is `fit_pgam.py`. Example:
```bash
python fit_pgam.py \
    --data_dir /path/to/data \
    --save_dir /path/to/results \
    --mouse 6000 \
    --session 1 \
    --window_size 0.01 \
    --window_step 0.01 \
    --use_units good/mua \
    --order 4 \
    --frac_eval 0.2
```

For batch processing on a SLURM cluster, see `submit_jobs.sh`.

## Repository Structure

- `core.py` – functions to compute spike rates and align neural and behavioral data.
- `fit_pgam.py` – preprocessing and model fitting script with command-line interface.
- `src/PGAM/` – PGAM library modules.
- `utils/` – helper utilities.
- `analysis.ipynb` – example analysis notebook.

