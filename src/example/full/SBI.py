import torch
import os
import sys

import time

import random
import math

current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import zuko
from torch import Tensor
from torch.distributions import Transform
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast
import torch.optim as optim
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator



# Import your training class and any required components

from utilities.Gassmann import simulator_full5
from utilities.PlotHighD import plot_5d_corner


from sbi.inference import SNPE
from sbi.inference import NPE, FMPE 
from sbi.neural_nets import flowmatching_nn
from sbi.inference import NPSE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)
from torch.distributions import MultivariateNormal
from sbi.utils import BoxUniform, MultipleIndependent

from utilities.MLP import SimpleVectorFieldNet

from utilities.FlowMatchingEstimator import FlowMatchingEstimator


start_time = time.perf_counter()

SEED = 1234


random.seed(SEED)


np.random.seed(SEED)


torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

num_dim = 5# Define the prior dimensions

x_obs = torch.tensor([0.64704126, 0.61732611])  # Provided observed data

# Number of forward evaluations
num_simulations = 4000

# Uniform[0,10] on θ₀ and θ₁
low_u  = torch.tensor([0.0, 0.0])
high_u = torch.tensor([10.0, 10.0])
uniform1 = BoxUniform(low=low_u, high=high_u)
uniform2 = BoxUniform(low=low_u, high=high_u)

# Joint Gaussian on θ₂,θ₃,θ₄ with your means & stds
means = torch.tensor([8.5, 0.37, 44.8])
stds  = torch.tensor([0.3,  0.02, 0.8])
cov3  = torch.diag(stds**2)
gauss3 = MultivariateNormal(loc=means, covariance_matrix=cov3)

# Combine into one 5D prior
prior = MultipleIndependent([uniform2, gauss3])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define prior and sample from it
theta = prior.sample((num_simulations,)).to(device)
x_obs = x_obs.to(device)

x = simulator_full5(theta)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

low_u = torch.tensor([0.0, 0.0], device=device)
high_u = torch.tensor([10.0, 10.0], device=device)
uniform2 = BoxUniform(low=low_u, high=high_u)

means = torch.tensor([8.5, 0.37, 44.8], device=device)
stds = torch.tensor([0.3, 0.02, 0.8], device=device)
cov3 = torch.diag(stds**2)
gauss3 = MultivariateNormal(loc=means, covariance_matrix=cov3)

prior = MultipleIndependent([uniform2, gauss3])
prior.to(device)


# Train the neural network
trainer = NPE(prior, device=device)
trainer.append_simulations(theta, x).train()
posterior = trainer.build_posterior()



num_samples = 10000
data_dim = 2
n_obs = 100

# Create observed data within uncertainty
xs = x_obs.unsqueeze(0).expand(n_obs, -1) + 0.01 * torch.randn(
    n_obs, data_dim, device=x_obs.device
)

# Sample from the learnt posterior
samples = posterior.sample_batched((num_samples,), x=xs)
samples_2d = samples.reshape(-1, samples.shape[-1]).cpu().numpy()
save_path = "./src/example/full/results/sbi_cbar_time.png"

# Plotting
plot_5d_corner(samples_2d, save_path=save_path)

end_time = time.perf_counter()

print(f"Total runtime: {end_time - start_time:.2f} seconds")
