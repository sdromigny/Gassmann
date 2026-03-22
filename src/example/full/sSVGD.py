import numpy as np
import torch

import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
from utilities.NormFlows import *
from utilities.SVGDFunc import *
from utilities.PlotHighD import *
from utilities.Gassmann import *
import numpy as np

import time
start_time = time.perf_counter()

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set observed data and uncertainty
x_obs = np.array([0.64704126, 0.61732611], dtype=np.float64)
sigma = 0.01



# Set inversion parameters
num_particles = 100
n_theta = 5

xs = []
for _ in range(num_particles):
    main_params = np.random.uniform(0, 10, size=2)
    latent_params = np.array([
        np.random.normal(8.5, 0.3),   # G_frame
        np.random.normal(0.37, 0.02), # Porosity
        np.random.normal(44.8, 0.8)   # Rho_grain
    ])
    xs.append(np.concatenate([main_params, latent_params]))

xs = np.array(xs, dtype=np.float64)

# Set log likelihood definition
lnprob = lnprob_factory(
    forward_model=simulator_full5,
    d_obs=x_obs,
    sigma=sigma, 
    device=device
)


svgd = sSVGD(
    lnprob=lnprob,
    kernel="rbf",
    h=1.0,
    weight="grad",
    out="samples_time.hdf5",
)

n_iter = 200000

step_sz = 1e-3
burn_in = 20000

thin = 10

z0 = np.random.randn(num_particles, n_theta)


losses, theta_final = svgd.sample(
    x0=z0,
    n_iter=n_iter,
    stepsize=step_sz,
    burn_in=burn_in,
    thin=thin,
    chunks=None
)

print("Final theta shape:", theta_final.shape)


import h5py
with h5py.File("samples_time.hdf5", "r") as f:
    z_samples = np.array(f["samples"])  

z_samples = z_samples.reshape(-1, n_theta)

# Map back to physical space
z_t = torch.tensor(z_samples, device=device)
theta_samples, _ = batch_constrained_transform(z_t)

theta_samples = theta_samples.detach().cpu().numpy()

# Save the samples and plot
np.save("./src/example/samples/full/ssvgd_larger_time.npy", theta_samples)

truths = [4.0, 7.0, 8.5, 0.37, 44.8]

plot_5d_corner(theta_samples, truths=truths, save_path="./src/example/full/results/ssvgd_larger_time.png")


end_time = time.perf_counter()

print(f"Total runtime: {end_time - start_time:.2f} seconds")
