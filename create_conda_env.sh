#!/bin/bash

# First update & install mamba (faster than conda)
conda update -n base -c defaults conda
conda install -c conda-forge mamba

# Create a conda environment for HydroBlocks
mamba create -n HBenv -y -c conda-forge
conda activate HBenv
mamba update --all -c conda-forge
mamba install -c conda-forge netcdf4 gdal geos jpeg scikit-learn numpy scipy h5py matplotlib cartopy mpi4py zarr opencv gfortran pandas numba
python -m pip install git+https://github.com/chaneyn/geospatialtools.git
python setup.py
