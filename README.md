# Single particle tracking

Functions for single particle tracking, diffusion analysis and off-rate analysis.


## Install instructions

You will need to install several packages in order to run this code. 
The easiest way to do this is with [Anaconda](https://docs.anaconda.com/anaconda/install/). 
Assuming Anaconda is already installed on your machine, you can set up an environment and install the necessary packages by running the following terminal commands in order:

    conda create -n spt python=3.5 anaconda
    conda activate spt
    conda install -c conda-forge trackpy
    pip install git+https://github.com/tsmbland/pims.git
    pip install tifffile==0.12.1
    pip install pims_nd2==1.1
    
The python code will now be ready to run. 
You can return to this environment in the future simply by running:

    conda activate spt
