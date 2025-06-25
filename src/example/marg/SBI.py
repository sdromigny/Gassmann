import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(src_path)


import numpy as np
import pints
import matplotlib.pyplot as plt

import random
import math
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

from utilities.Gassmann import simulator_prob, simulator_det, sample_nuis_parameters_numpy
from utilities.Histogram2d import pairplot


import torch

from sbi.inference import SNPE
from sbi.inference import NPE
from sbi.neural_nets import flowmatching_nn
from sbi.inference import NPSE
from sbi.utils import BoxUniform
from sbi.utils.user_input_checks import (
    check_sbi_inputs,
    process_prior,
    process_simulator,
)

from utilities.MLP import SimpleVectorFieldNet

from utilities.FlowMatchingEstimator import FlowMatchingEstimator

# num_dim = 2# Define the prior

# simulator=simulator_prob

# prior = BoxUniform(low=0.01 * torch.ones(num_dim), high=10 * torch.ones(num_dim))

# # Check prior, return PyTorch prior.
# prior, num_parameters, prior_returns_numpy = process_prior(prior)

# # Check simulator, returns PyTorch simulator able to simulate batches.
# simulator = process_simulator(simulator, prior, prior_returns_numpy)

# # Consistency check after making ready for sbi.
# check_sbi_inputs(simulator, prior)

# # Sample from the prior and simulate
# num_simulations = 2000
# theta = prior.sample((num_simulations,))
# x = simulator(theta)
# print(x)

# theta_true = torch.tensor([[4, 7]])
# # generate our observation
# x_obs = torch.tensor([0.64704126, 0.61732611])  # Provided observed data

# trainer = NPE(prior)
# trainer.append_simulations(theta, x).train()
# posterior = trainer.build_posterior()

# samples = posterior.sample((10000,), x=x_obs)


# save_path = "/home/users/scro4690/Documents/GenInv/SBIcompare/src/plotting/figures/Gassmann/sbi_prob.png"

# m_true=torch.tensor([4, 7])


#pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)

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
num_epochs = 20000
batch_size = 128
dim = 2
simulator = simulator_prob  # or simulator_det

# Define vector field net and embedding net
hidden_units = 12
vector_field_net = SimpleVectorFieldNet(
    input_dim=dim,
    condition_dim=dim,
    time_encoding_dim=6,
    hidden_dim=hidden_units
).to(device)

embedding_net = nn.Identity().to(device)

# Initialize estimator
estimator = FlowMatchingEstimator(
    net=vector_field_net,
    input_shape=torch.Size([dim]),
    condition_shape=torch.Size([dim]),
    embedding_net=embedding_net,
).to(device)

# Optimizer
optimizer = optim.Adam(estimator.parameters(), lr=1e-3)

# Loss history
loss_hist = np.array([])
best_loss = float('inf')
best_model_state = None

# Training loop
for it in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    x1_dist = torch.distributions.Uniform(0.01, 10.0)
    x1 = x1_dist.sample((batch_size, dim)).to(device)

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
num_samples = 10000
eval_batch_size = 25

d_obs = torch.tensor([0.64704126, 0.61732611], device=device)
d_pdf = d_obs + 0.01 * torch.randn(num_samples, dim, device=device)

samples = []
with torch.no_grad():
    for i in range(0, num_samples, eval_batch_size):
        batch_d_test = d_pdf[i: i + eval_batch_size]
        flow = estimator.flow(batch_d_test)
        batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]])).view(-1, dim)
        samples.append(batch_samples)

samples = torch.cat(samples, dim=0)

# Plot
samples_np = samples.cpu().numpy()

#np.save("/home/users/scro4690/Documents/GenInv/SBIcompare/src/examples/gassmann/samples/fmpe_samples_prob.npy",samples_np)
m_true = torch.tensor([4, 7]).numpy()

save_path = "/home/users/scro4690/Documents/GenInv/SBIcompare/src/plotting/figures/Gassmann/fm_prob_test.png"
pairplot(samples_np, m_true, fontsize=15, save_path=save_path)




###################################################################################################################



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
num_epochs = 20000
batch_size = 128
dim = 2
simulator = simulator_det  # or simulator_det

# Define vector field net and embedding net
hidden_units = 12
vector_field_net = SimpleVectorFieldNet(
    input_dim=dim,
    condition_dim=dim,
    time_encoding_dim=6,
    hidden_dim=hidden_units
).to(device)

embedding_net = nn.Identity().to(device)

# Initialize estimator
estimator = FlowMatchingEstimator(
    net=vector_field_net,
    input_shape=torch.Size([dim]),
    condition_shape=torch.Size([dim]),
    embedding_net=embedding_net,
).to(device)

# Optimizer
optimizer = optim.Adam(estimator.parameters(), lr=1e-3)

# Loss history
loss_hist = np.array([])
best_loss = float('inf')
best_model_state = None

# Training loop
for it in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    x1_dist = torch.distributions.Uniform(0.01, 10.0)
    x1 = x1_dist.sample((batch_size, dim)).to(device)

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
num_samples = 10000
eval_batch_size = 25

d_obs = torch.tensor([0.64704126, 0.61732611], device=device)
d_pdf = d_obs + 0.01 * torch.randn(num_samples, dim, device=device)

samples = []
with torch.no_grad():
    for i in range(0, num_samples, eval_batch_size):
        batch_d_test = d_pdf[i: i + eval_batch_size]
        flow = estimator.flow(batch_d_test)
        batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]])).view(-1, dim)
        samples.append(batch_samples)

samples = torch.cat(samples, dim=0)

# Plot
samples_np = samples.cpu().numpy()

#np.save("/home/users/scro4690/Documents/GenInv/SBIcompare/src/examples/gassmann/samples/fmpe_samples_det.npy",samples_np)
m_true = torch.tensor([4, 7]).numpy()

save_path = "/home/users/scro4690/Documents/GenInv/SBIcompare/src/plotting/figures/Gassmann/fm_det_test.png"
pairplot(samples_np, m_true, fontsize=15, save_path=save_path)