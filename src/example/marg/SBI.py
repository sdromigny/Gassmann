import torch
import os
import sys

current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import pints
import matplotlib.pyplot as plt
import torch.nn as nn
import random
import math
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import zuko
from torch import Tensor
from torch.distributions import Transform
from zuko.distributions import DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast
import torch.optim as optim

from utilities.Gassmann import simulator_prob, simulator_det, sample_nuis_parameters_numpy
from utilities.Histogram2d import pairplot


import torch

from utilities.MLP import SimpleVectorFieldNet

from utilities.FlowMatchingEstimator import FlowMatchingEstimator

########################################################################################################################
### Probabilistic version ###


import time
start_time = time.perf_counter()

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config
num_epochs = 20000
batch_size = 10
dim = 2

# Set forward model
simulator = simulator_prob  

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
loss_hist = []
best_loss = float('inf')
best_model_state = None

# Training loop
for it in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    x1 = 0.01 + (10.0 - 0.01) * torch.rand(batch_size, dim, device=device)

    x0,_ = simulator(x1)
    x0=x0.to(device)
    loss = estimator.loss(x1, x0).mean()

    loss.backward()
    optimizer.step()

    loss_hist.append(loss.detach().item())

loss_hist = np.asarray(loss_hist)
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
with torch.inference_mode():
    for i in range(0, num_samples, eval_batch_size):
        batch_d_test = d_pdf[i: i + eval_batch_size]
        flow = estimator.flow(batch_d_test)
        batch_samples = flow.sample(torch.Size([batch_d_test.shape[0]])).view(-1, dim)
        samples.append(batch_samples)

samples = torch.cat(samples, dim=0)

# Plot
samples_np = samples.cpu().numpy()

np.save("./src/example/samples/marg/prob/fmpe_samples_prob_small_time.npy",samples_np)
m_true = torch.tensor([4, 7]).numpy()

save_path = "./src/example/marg/results/fm_prob_small_time.png"
pairplot(samples_np, m_true, fontsize=15, save_path=save_path)

end_time = time.perf_counter()
print(f"Total runtime: {end_time - start_time:.2f} seconds")




###################################################################################
### Deterministic version ###
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

current_dir = os.path.dirname(__file__)
src_path = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(src_path)

from sbi.inference import NPE
from sbi.utils import BoxUniform
from utilities.Gassmann import simulator_det
from utilities.Histogram2d import pairplot

start_time = time.perf_counter()

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define observed data values
x_obs = torch.tensor([0.64704126, 0.61732611], device=device)

# Define prior
prior = BoxUniform(
    low=torch.tensor([0.0, 0.0], device=device),
    high=torch.tensor([10.0, 10.0], device=device),
)
prior.to(device)

# Number of forward evaluations
num_simulations = 7000
theta = prior.sample((num_simulations,))

x = simulator_det(theta)
if isinstance(x, tuple):
    x = x[0]
x = x.to(device)

print("theta shape:", theta.shape)
print("x shape:", x.shape)

# Training the neural networks
trainer = NPE(prior, device=device)
density_estimator = trainer.append_simulations(theta, x).train()
posterior = trainer.build_posterior(density_estimator)

# Sample the posterior
num_samples = 100000
n_obs = 100
sigma_obs = 0.01

# Noisy observations to take into account uncertainty
xs = x_obs.unsqueeze(0).expand(n_obs, -1) + sigma_obs * torch.randn(
    n_obs, x_obs.numel(), device=device
)


samples = posterior.sample_batched((num_samples,), x=xs)


samples_2d = samples.reshape(-1, samples.shape[-1]).cpu().numpy()


# Plotting
m_true = np.array([4, 7])
save_path = "./src/example/marg/results/sbi_det_time.png"
pairplot(samples_2d, m_true, fontsize=15, save_path=save_path)

end_time = time.perf_counter()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
