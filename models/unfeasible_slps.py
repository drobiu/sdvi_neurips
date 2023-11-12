from ctypes import addressof
import torch
import matplotlib.pyplot as plt
import math
import logging
import copy
import numpy as np
import seaborn as sns
import torch.nn as nn

sns.reset_orig()

from pyro.distributions import Normal, Uniform
from pyro import plate, sample


class UnfeasibleSLPs(nn.Module):
    autoguide_hide_vars = []

    does_lppd_evaluation = False

    slps_identified_by_discrete_samples = False

    def __init__(self):
        super().__init__()

    # 1: bool earthquake, burglary, alarm,
    # phoneWorking, maryWakes, called;
    # 2: earthquake = Bernoulli(0.0001);
    # 3: burglary = Bernoulli(0.001);
    # 4: alarm = earthquake or burglary;
    # 5: if (earthquake)
    # 6:    phoneWorking = Bernoulli(0.7);
    # 7: else
    # 8: phoneWorking = Bernoulli(0.99);
    # 9: if(alarm) {
    # 10:   if(earthquake)
    # 11:       maryWakes = Bernoulli(0.8);
    # 12:   else
    # 13:       maryWakes = Bernoulli(0.6);
    # 14: } else
    # 15:   maryWakes = Bernoulli(0.2);
    # 16: called = maryWakes and phoneWorking;
    # 17: observe(called);
    # 18: return burglary;

    def __call__(self):
        earthquake = sample("earthquake", Uniform(0, 1)) < 0.5
        burglary = sample(f"e{earthquake}_burglary", Uniform(0, 1)) < 0.5
        c = sample(f"e{earthquake}_b{burglary}", Uniform(0, 1)) < 0.5

        alarm = earthquake | burglary

        if earthquake:
            phoneWorking = sample("phoneWorking1", Uniform(0, 1), infer={'branching': True}) < 0.7
        else:
            phoneWorking = sample("phoneWorking2", Uniform(0, 1), infer={'branching': True}) < 0.99

        if alarm:
            if earthquake:
                maryWakes = sample("maryWakes1", Uniform(0, 1), infer={'branching': True}) < 0.8
            else:
                maryWakes = sample("maryWakes2", Uniform(0, 1), infer={'branching': True}) < 0.6
        
        else:
            maryWakes = sample("maryWakes3", Uniform(0, 1), infer={'branching': True}) < 0.2

        called = maryWakes & phoneWorking

        return called


    def calculate_ground_truth_weights(self, sdvi):
        return None, None

    def make_parameter_plots(self, results, guide, address_trace, file_prefix):
        # All address traces have the form "u,x_{prior_mean}"
        return

    def plot_posterior_samples(self, posterior_samples, fname):
        logging.info("Not doing a posterior plot for this model.")

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        return torch.tensor(float("nan"))

