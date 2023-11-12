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


class LowLikelihoodSLPs(nn.Module):
    autoguide_hide_vars = []

    does_lppd_evaluation = False

    slps_identified_by_discrete_samples = False

    def __init__(self, if_probability=0.5, slp_num_power=2):
        super().__init__()
        self.if_probability = if_probability
        self.slp_num_power = slp_num_power

    def __call__(self):
        m1 = sample("m1", Uniform(0, 1))

        std = 1
        mu = 0

        if m1 < self.if_probability:
            slp = f"{m1 / (self.if_probability * 10):.{self.slp_num_power}f}".split('.')[1] # Generate 100 branch slp ids
            y = sample(f"y_{slp}", Normal(mu, std), infer={'branching': True})

        else:
            y = sample("y_else", Normal(mu, std), infer={'branching': True})

        return y


    def calculate_ground_truth_weights(self, sdvi):
        return None, None

    def make_parameter_plots(self, results, guide, address_trace, file_prefix):
        # All address traces have the form "u,x_{prior_mean}"
        return

    def plot_posterior_samples(self, posterior_samples, fname):
        logging.info("Not doing a posterior plot for this model.")

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        return torch.tensor(float("nan"))

