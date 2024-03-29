# Recurrent predictive coding networks for associative memory employing covariance learning

## 1. Description
This repository contains code to perform experiments with recurrent predictive coding networks on associative memory tasks.

The preprint associated with the code repository can be found [here](https://www.biorxiv.org/content/10.1101/2022.11.09.515747v1.abstract).

## 2. Installation
To run the code, you should first install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://conda.io/miniconda.html) (preferably the latter), 
and then clone this repository to your local machine.

Once these are installed and cloned, you can simply use the appropriate `.yml` file to create a conda environment. 
For Ubuntu or Mac OS, open a terminal, go to the repository directory; for Windows, open the Anaconda Prompt, and then enter:

1. `conda env create -f environment.yml`  
2. `conda activate cov-env`
3. `pip install -e .`  

## 3. Use
Once the above are done, you can simply run a script by entering for example:

`python scripts/single_layer_PCNs.py`

A directory named `results` will the be created to store all the data and figures collected from the experiments.

A jupyter notebook will be produced to show how to generate figure from our paper based on the collected results (Coming Soon).
