# HydroBlocks

## Install dependencies

HydroBlocks relies on a number python libraries. To make this straightforward use conda (http://conda.pydata.org/miniconda.html).

The easiest way to do this on *nix is by using the "create_conda_env.sh" script in this repository. 
This will create a conda environment called "HBenv" with all the necessary libraries.

Alternatively, if you'd like to install the dependencies yourself, you can do so by following the instructions.
First make certain that the base environment is up to date and install mamba (much faster drop-in replacement for conda):

```bash
conda update -n base -c defaults conda
conda install -c conda-forge mamba
```

Then create a new environment called "HBenv" and install the dependencies:

```bash
mamba create -n HBenv -y -c conda-forge python=3.11
conda activate HBenv
mamba update --all -c conda-forge
mamba install -c conda-forge netcdf4 gdal geos jpeg scikit-learn numpy scipy h5py matplotlib cartopy mpi4py zarr opencv gfortran pandas numba
conda install --force-reinstall proj
python -m pip install git+https://github.com/chaneyn/geospatialtools.git
```

The reinstallation of the proj package is sometimes necessary so that the environment can find the proj-database files.


## Setup HydroBlocks

Next, clone and install HydroBlocks either using your favorite git client or by running the following commands:

```bash
git clone -b dev_errorMsg https://github.com/LeoTolstoi/HydroBlocks/
cd HydroBlocks
python setup.py
```

## Run Example Model

Run an example model using the following commands, starting out from the HydroBlocks directory:

```bash
cd ..
wget https://www.dropbox.com/s/w10u8ocghk3oe21/HB_sample_nv.tar.gz?dl=0
tar -xvzf HB_sample_nv.tar.gz
cd HB_sample
python ../HydroBlocks/Preprocessing/Driver.py metadata.json
python ../HydroBlocks/HydroBlocks/Driver.py metadata.json 
```

## Plot Results 

The output of the example model can be plotted using the following commands:

```bash
python plot.py
```

