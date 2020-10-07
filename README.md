HydroBlocks
==========

The following steps walk through how to install HydroBlocks and run it over the SGP site in Oklahoma

**1. Create a conda environment named HB from the yml file. Note that the only current yml file in the repository is for a linux64 machine.** 

```
conda update conda
conda env create -f yml/HB_linux64.yml
source activate HB
```

**2. Clone and install the dev_nate branch of HydroBlocks.**

```
git clone --single-branch --branch dev_nate https://github.com/chaneyn/HydroBlocks.git
cd HydroBlocks
python setup.py 
cd ..
```

**3. Download the SGP site data and run the model.**

```
wget http://hydrology.cee.duke.edu/HydroBlocks/SGP_OK_1deg.tar.gz
tar -xvzf SGP_OK_1deg.tar.gz
vi SGP_OK_1deg/experiments/json/baseline.json
Set the variable rdir to the absolute path of SGP_OK_1deg
mpirun -n 16 ./HydroBlocks/HB -m SGP_OK_1deg/experiments/json/baseline.json -t preprocess
mpirun -n 16 ./HydroBlocks/HB -m SGP_OK_1deg/experiments/json/baseline.json -t model
```

