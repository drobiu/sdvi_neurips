# Code for "Rethinking Variational Inference for Probabilistic Programs with Stochastic Support"

## Installation and setup

This repository is using [Poetry](https://python-poetry.org/) to manage dependencies. 
To run code you therefore first have to install Poetry on your system.

Then to install all the dependencies run
```bash
poetry install
```
and then you should be good to go.

## Outline of repository

- `data`
    - `data/airline` contains the data for the GP experiment
    - `data/gmm` contains the data for the GMM experiment
- `models` contains the implementations of the Pyro models
    - `models/pyro_extensions/` contains our implementation of SDVI as well as our Pyro implementation of DCC
- `notebooks/paper_figures.ipynb` contains the code to reproduce the figures in the paper 
- This repository uses the [Hydra](https://hydra.cc/) configuration management systems to configure experiments. There are several configuration directories for the different methods.
    - `conf_pyro_extension` contains the configurations for SDVI

For the models that require access to data stored on disk require an absolute path in their configuration file. 
For anonymization these absolute paths have been removed and you will manually have to enter these paths for your local machine.
The necessary places in the configuration files are marked with `TODO`s.


## Running experiments

## Distinct SLPs
python run_exp_pyro_extension.py name=distinct_slps_sdvi sdvi.forward_kl_iter=100 sdvi.forward_kl_num_particles=50 sdvi.elbo_estimate_num_particles=10 sdvi.exclusive_kl_num_particles=5 model=distinct_slps resource_allocation=successive_halving resource_allocation.num_total_iterations=20

## Overlapping SLPs
python run_exp_pyro_extension.py name=overlapping_slps_sdvi sdvi.forward_kl_iter=100 sdvi.forward_kl_num_particles=50 sdvi.elbo_estimate_num_particles=100 sdvi.exclusive_kl_num_particles=5 model=slps_with_overlap resource_allocation=successive_halving resource_allocation.num_total_iterations=200 

## Low Likelihood SLPs
python run_exp_pyro_extension.py name=low_likelihood_slps_sdvi_1000 sdvi.forward_kl_iter=100 sdvi.forward_kl_num_particles=1000 sdvi.elbo_estimate_num_particles=5 sdvi.exclusive_kl_num_particles=5 model=low_likelihood_slps resource_allocation=successive_halving resource_allocation.num_total_iterations=50 sdvi.find_slp_samples=10000 sdvi.save_metrics_every_n=5

## Unfeasible SLPs
python run_exp_pyro_extension.py name=unfeasible_slps sdvi.forward_kl_iter=100 sdvi.forward_kl_num_particles=1000 sdvi.elbo_estimate_num_particles=5 sdvi.exclusive_kl_num_particles=5 model=unfeasible_slps resource_allocation=successive_halving resource_allocation.num_total_iterations=50 sdvi.find_slp_samples=1000 sdvi.save_metrics_every_n=5