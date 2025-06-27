import torch
import os
import sys


import numpy as np
import pints
import matplotlib.pyplot as plt

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
import os
import torch
import torch.nn as nn
import zuko
from torch import Tensor
from torch.distributions import Transform
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast
import torch.optim as optim
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator
#from sbi.neural_nets.estimators.flowmatching_estimator import (FlowMatchingEstimator)
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

# Import your training class and any required components

from utilities.Gassmann import simulator_full5
from utilities.PlotHighD import plot_5d_corner


import torch

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

num_dim = 5# Define the prior

x_obs = torch.tensor([0.64704126, 0.61732611])  # Provided observed data

# Sample from the prior and simulate
num_simulations = 5000
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
print(x)


# generate our observation

trainer = NPE(prior)
trainer.append_simulations(theta, x).train()
posterior = trainer.build_posterior()

num_samples=100000
data_dim=2
xs=x_obs + 0.01 * torch.randn(1000, data_dim)
samples = posterior.sample_batched((num_samples,), x=xs)


save_path = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/full/results/sbi_fmpe.png"


plot_5d_corner(samples, save_path=save_path)

########################################################################################################################


# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
num_epochs = 30000
batch_size = 200
input_dim = 5
output_dim=2
simulator = simulator_full5  # or simulator_det
x_obs = torch.tensor([0.64704126, 0.61732611], device=device)
# Define vector field net and embedding net
hidden_units = 20
vector_field_net = SimpleVectorFieldNet(
    input_dim=input_dim,
    condition_dim=output_dim,
    time_encoding_dim=6,
    hidden_dim=hidden_units
).to(device)

embedding_net = nn.Identity().to(device)

# Initialize estimator
estimator = FlowMatchingEstimator(
    net=vector_field_net,
    input_shape=torch.Size([input_dim]),
    condition_shape=torch.Size([output_dim]),
    embedding_net=embedding_net,
).to(device)

# Optimizer
optimizer = optim.Adam(estimator.parameters(), lr=1e-4)

# Loss history
loss_hist = np.array([])
best_loss = float('inf')
best_model_state = None

# Training loop
for it in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    xs = []
    for _ in range(batch_size):
        main_params = np.random.uniform(0, 10, size=len(x_obs))
        latent_params = np.array([
            np.random.normal(8.5, 0.3),  # G_frame
            np.random.normal(0.37, 0.02),  # Porosity
            np.random.normal(44.8, 0.8)   # Rho_grain
        ])
        xs.append(np.concatenate([main_params, latent_params]))
    x1 = torch.tensor(xs, dtype=torch.float32, device=device)

    x0 = simulator(x1).to(device)

    loss = estimator.loss(x1, x0).mean()

    loss.backward()
    optimizer.step()

    loss_hist = np.append(loss_hist, loss.detach().cpu().numpy())

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Posterior sampling
num_samples = 100000
eval_batch_size = 25


d_pdf = x_obs + 0.01 * torch.randn(num_samples, output_dim, device=device)

samples = []
with torch.no_grad():
    for i in range(0, num_samples, eval_batch_size):
        batch_d_test = d_pdf[i: i + eval_batch_size]
        flow = estimator.flow(batch_d_test)
        batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]])).view(-1, input_dim)
        samples.append(batch_samples)

samples = torch.cat(samples, dim=0)

# Plot
samples_np = samples.cpu().numpy()

#np.save("/home/users/scro4690/Documents/GenInv/SBIcompare/src/examples/gassmann/samples/fmpe_samples_prob.npy",samples_np)


save_path = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/full/results/fm.png"
plot_5d_corner(samples_np,save_path=save_path)


