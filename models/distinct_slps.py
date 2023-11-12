from ctypes import addressof
import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import math
import logging
import copy
import numpy as np
import seaborn as sns
import torch.nn as nn

sns.reset_orig()

from .abstract_model import AbstractModel


class DistinctSLPs(nn.Module):
    autoguide_hide_vars = []

    does_lppd_evaluation = False

    slps_identified_by_discrete_samples = False

    def __init__(self, data_path):
        super().__init__()
        self.data = self.load_data(data_path)
        self.batch_size = 10

    def __call__(self):
        x = pyro.sample("x", dist.Normal(0, 1))
        m1 = pyro.sample("m1", dist.Uniform(0, 1))

        if m1 < 0.5:
            std = 0.62177
            with pyro.plate(f"data1", len(self.data), subsample_size=self.batch_size) as ind:
                # print(ind)
                batch = self.data[ind]
                y = pyro.sample("y1", dist.Normal(x, std), obs=batch, infer={'branching': True})

        else:
            std = 2.0
            with pyro.plate(f"data2", len(self.data), subsample_size=self.batch_size) as ind:
                batch = self.data[ind]
                y = pyro.sample("y2", dist.Normal(x, std), obs=batch, infer={'branching': True})

        return x


    @staticmethod
    def load_data(data_path):
        # data = torch.tensor(np.loadtxt(data_path))
        n_samples = 200
        data = torch.tensor([np.random.normal() for _ in range(n_samples)])

        return data


    def calculate_ground_truth_weights(self, sdvi):
        return None, None

    def make_parameter_plots(self, results, guide, address_trace, file_prefix):
        # All address traces have the form "u,x_{prior_mean}"
        return

    def plot_posterior_samples(self, posterior_samples, fname):
        logging.info("Not doing a posterior plot for this model.")

    def evaluation(self, posterior_samples, ground_truth_weights=None):
        return torch.tensor(float("nan"))