import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import math
import matplotlib.pyplot as plt
import normflows as nf
from tqdm import tqdm
from utilities.NormFlows import *
import numpy as np
import matplotlib.pyplot as plt
from utilities.Gassmann import simulator_full5, sample_and_log_gaussians  # Or simulator_det
from utilities.PlotHighD import plot_5d_corner

import time
start_time = time.perf_counter()
# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Flow model setup
K = 1
dim = 5
flows = [nf.flows.Planar((dim,)) for _ in range(K)]
q0 = nf.distributions.DiagGaussian(dim)
nfm = Linear(dim=dim, kernel='fullrank').to(device)


# Training config
num_epochs = 1000
batch_size =10

# Number of forward evaluations = num_epochs*batch_size

# Set observed data and uncertainty
sigma = 0.01
observed_data = torch.tensor([0.64704126, 0.61732611], device=device)

# Define forward model
simulator=simulator_full5

loss_hist = np.array([])
best_loss = float('inf')
best_model_state = None



optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-3)

for it in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    z0 = torch.randn(batch_size, dim, device=device)
    w, log_det_flow = nfm.forward_and_log_det(z0)      


    theta, log_det_trans = batch_constrained_transform(w)

    log_det = log_det_flow + log_det_trans            

    synthetic_value = simulator(theta)              

    log_p = -0.5 * (((synthetic_value - observed_data) / sigma) ** 2).sum(dim=1)
    log_q0 = -0.5 * torch.sum(z0**2, dim=1) - 0.5 * dim * math.log(2*math.pi)
    log_q  = log_q0 - log_det
    log_pn = sample_and_log_gaussians(theta[:, 2:] )

    loss = -(log_q + log_p + log_pn).mean()
    loss.backward()
    optimizer.step()

    loss_hist = np.append(loss_hist, loss.item())

    if loss < best_loss:
        best_loss = loss
        best_model_state = nfm.state_dict()
        print(f"Model saved at iteration {it} with loss: {best_loss}")

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, label='loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss for VI Normalising Flows")


# Sampling and plotting
z, _ = nfm.sample(num_samples=100000)  

# bounds for Uniform dims 0–1
u_lo, u_hi = 0.0, 10.0

# Gaussian params for dims 2,3,4
gauss_means = [8.5, 0.37, 44.8]
gauss_stds  = [0.3,  0.02, 0.8]

# Map back to physical space
theta_samples = unconstrained_to_constrained_5d(
    z, u_lo, u_hi,
    gauss_means, gauss_stds
)

print(theta_samples.shape)  

theta_samples=theta_samples.detach().cpu().numpy()

save_path = "./src/example/full/results/nf_large_cbar_time"

# Save the samples and plot

np.save("./src/example/samples/full/nf_large_time.npy",theta_samples)
truths = [4.0, 7.0, 8.5, 0.37, 44.8]

plot_5d_corner(theta_samples, truths=truths,save_path=save_path)

end_time = time.perf_counter()

print(f"Total runtime: {end_time - start_time:.2f} seconds")
