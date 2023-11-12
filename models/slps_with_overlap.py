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


class SLPsWithOverlap(nn.Module):
    autoguide_hide_vars = []

    does_lppd_evaluation = False

    slps_identified_by_discrete_samples = False

    def __init__(self,):
        super().__init__()
        self.data = self.load_data()
        self.batch_size = 10

    def __call__(self):
        theta1 = pyro.sample("theta1", dist.Normal(0, 1))
        theta2 = pyro.sample("theta2", dist.Normal(0, 1))
        m1 = pyro.sample("m1", dist.Uniform(0, 1))

        std = 1

        if m1 < 0.5:
            with pyro.plate(f"data1", len(self.data), subsample_size=self.batch_size) as ind:
                batch = self.data[ind]
                mu = theta1 * batch[:, 0] + theta2 * batch[:, 2]

                y = pyro.sample("y1", dist.Normal(mu, std), obs=batch[:, 4], infer={'branching': True})

        else:
            with pyro.plate(f"data2", len(self.data), subsample_size=self.batch_size) as ind:
                batch = self.data[ind]
                mu = theta1 * batch[:, 0] + theta2 * batch[:, 3]

                y = pyro.sample("y2", dist.Normal(mu, std), obs=batch[:, 4], infer={'branching': True})

        return y


    @staticmethod
    def load_data():
        beta = [1.5, 1.5, 0.3, 0.1]

        def generate_data(n):
            data = []
            
            for i in range(n):
                y = 0
                row = []
                e = np.random.normal()

                for b in beta:
                    x_current = np.random.normal()
                    row.append(x_current)
                    y += b * x_current
                
                row.append(y + e)
                data.append(row)
                
            return np.array(data)

        n_samples = 200
        data = torch.tensor(generate_data(n_samples))

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