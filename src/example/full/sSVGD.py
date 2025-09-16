
import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import torch
from utilities.SVGDFunc import FullsSVGD
from utilities.PlotHighD import *


# Define observed data and noise
x_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
sigma = 0.01

# Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Number of particles & initial positions in θ‐space
num_particles = 100
n_theta=5
xs = []
for _ in range(num_particles):
    main_params = np.random.uniform(0, 10, size=len(x_obs))
    latent_params = np.array([
        np.random.normal(8.5, 0.3),  # G_frame
        np.random.normal(0.37, 0.02),  # Porosity
        np.random.normal(44.8, 0.8)   # Rho_grain
    ])
    xs.append(np.concatenate([main_params, latent_params]))
xs = np.array(xs)

# Instantiate SVGD sampler
svgd = FullsSVGD(d_obs=x_obs, sigma=sigma, device=device)

# Run SVGD
n_iter  = 10000000
step_sz = 1e-3
bandw   = -1   # median trick
alpha   = 0.9

# If you want to track every iteration’s particles:
particle_history = svgd.update(
    x0=xs,
    n_iter=n_iter,
    stepsize=step_sz,
    bandwidth=bandw,
    alpha=alpha,
    debug=True,
    track_history=True,
)
print("particle_history.shape:", particle_history.shape)


# Discard burn‐in and flatten
burn_in = 10000
thin=10
chains = particle_history[burn_in:, :, :]  # shape = (n_iter+1 - burn_in, num_particles, n_theta)

chains_thinned = chains[::thin, :, :] 


samples = chains_thinned.reshape(-1, n_theta)      # shape = ((n_iter+1 - burn_in)*num_particles, n_theta)
print("Flattened samples shape:", samples.shape)

#Plot pairplot of samples


save_path = "src/example/full/results/ssvgd.png"

m_true=torch.tensor([4, 7])

samples=np.vstack(chains)

np.save("src/example/samples/ssvgd.npy",samples)

print(samples.shape)

plot_5d_corner(samples, save_path=save_path)