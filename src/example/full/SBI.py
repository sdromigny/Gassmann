
import os
import sys


import random

current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import matplotlib.pyplot as plt



# Import your training class and any required components

from utilities.Gassmann import simulator_full5
from utilities.PlotHighD import plot_5d_corner


import torch


from sbi.inference import NPE

from torch.distributions import MultivariateNormal
from sbi.utils import BoxUniform, MultipleIndependent

from utilities.MLP import SimpleVectorFieldNet

from utilities.FlowMatchingEstimator import FlowMatchingEstimator

# 1) Pick a seed:
SEED = 1234

# 2) Python built-in RNG
random.seed(SEED)

# 3) NumPy RNG
np.random.seed(SEED)

# 4) PyTorch RNG (both CPU and CUDA)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 5) CUDNN backend settings for determinism (may slow you down slightly)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

num_dim = 5# Define the prior

x_obs = torch.tensor([0.64704126, 0.61732611])  # Provided observed data

# Sample from the prior and simulate
num_simulations = 4000
# 1) Uniform[0,10] on θ₀ and θ₁
low_u  = torch.tensor([0.0, 0.0])
high_u = torch.tensor([10.0, 10.0])
uniform1 = BoxUniform(low=low_u, high=high_u)
uniform2 = BoxUniform(low=low_u, high=high_u)

# 2) Joint Gaussian on θ₂,θ₃,θ₄ with your means & stds
means = torch.tensor([8.5, 0.37, 44.8])
stds  = torch.tensor([0.3,  0.02, 0.8])
cov3  = torch.diag(stds**2)
gauss3 = MultivariateNormal(loc=means, covariance_matrix=cov3)

# 3) Combine into one 5D prior
prior = MultipleIndependent([uniform2, gauss3])

# Test drawing a sample
theta = prior.sample((num_simulations,))

x = simulator_full5(theta)



# generate our observation

trainer = NPE(prior)
trainer.append_simulations(theta, x).train()
posterior = trainer.build_posterior()

num_samples=100000
data_dim=2
xs=x_obs + 0.01 * torch.randn(100, data_dim)
samples = posterior.sample_batched((num_samples,), x=xs)

samples_2d = samples.reshape(-1, samples.shape[-1])
save_path = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/full/results/sbi.png"

np.save("/home/users/scro4690/Documents/GenInv/Gassmann/src/example/samples/full/sbi.npy",samples_2d)
plot_5d_corner(samples_2d, save_path=save_path)


