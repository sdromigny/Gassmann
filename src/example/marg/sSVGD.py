
import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import torch
from utilities.SVGDFunc import SVGDGassmannDet, SVGDGassmannProb, sSVGDGassmannProb, sSVGDGassmannDet  # the class file you wrote above
from utilities.Histogram2d import pairplot

# 1) Set up noise & observed data
d_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
sigma = 0.01

# 2) Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 3) Number of particles and initial positions
num_particles = 100
n_params      = 2     # e.g., if simulator_det expects 4 parameters
x0 = np.random.uniform(0, 10, size=(num_particles, n_params)).astype(np.float32)

# 4) Instantiate SVGD, passing in d_obs, sigma, device
svgd = SVGDGassmannDet(d_obs=d_obs, sigma=sigma, device=device)


particle_history = svgd.update(
    x0=x0,
    n_iter=10000,
    stepsize=1e-3,
    bandwidth=-1,
    alpha=0.9,
    debug=True,
    track_history=True
)
print(particle_history.shape)


# Define the burn-in period
burn_in = 1000  # Adjust this value based on your needs


# drop the first `burn_in` iterations along axis=0
chains = particle_history[burn_in :, :, :]
# Suppose chains.shape == (N_iter_after_burn, num_particles, n_params)
_, num_particles, n_params = chains.shape

# Method 1: reshape
samples = chains.reshape(-1, n_params)  # shape = ((N_iter_after_burn * num_particles), n_params)


save_path = "src/example/marg/results/svgd_det.png"

m_true=torch.tensor([4, 7])


print(samples.shape)

pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)

####################################################################################################################################


# 1) Define observed data and noise
d_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
sigma = 0.01

# 2) Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 3) Number of particles & initial positions in θ‐space
num_particles = 100
n_theta = 2  # e.g. if θ has 2 components
x0 = np.random.uniform(0.0, 10.0, size=(num_particles, n_theta)).astype(np.float32)

# 4) Instantiate SVGD sampler
svgd = sSVGDGassmannProb(d_obs=d_obs, sigma=sigma, device=device)

# 5) Run SVGD
n_iter  = 5000
step_sz = 1e-3
bandw   = -1   # median trick
alpha   = 0.9

# If you want to track every iteration’s particles:
particle_history = svgd.update(
    x0=x0,
    n_iter=n_iter,
    stepsize=step_sz,
    bandwidth=bandw,
    alpha=alpha,
    debug=True,
    track_history=True,
)
print("particle_history.shape:", particle_history.shape)
# → (n_iter+1, num_particles, n_theta)

# 6) Discard burn‐in and flatten
burn_in = 1000
chains = particle_history[burn_in:, :, :]  # shape = (n_iter+1 - burn_in, num_particles, n_theta)
samples = chains.reshape(-1, n_theta)      # shape = ((n_iter+1 - burn_in)*num_particles, n_theta)
print("Flattened samples shape:", samples.shape)

# 7) (Optional) Plot pairplot of θ‐samples

save_path = "src/example/marg/results/ssvgd_prob.png"

m_true=torch.tensor([4, 7])

samples=np.vstack(chains)

np.save("src/example/samples/ssvgd_samples_prob.npy",samples)

pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)




##################################################################################################################



# 1) Define observed data and noise
d_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
sigma = 0.01

# 2) Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 3) Number of particles & initial positions in θ‐space
num_particles = 100
n_theta = 2  # e.g. if θ has 2 components
x0 = np.random.uniform(0.0, 10.0, size=(num_particles, n_theta)).astype(np.float32)

# 4) Instantiate SVGD sampler
svgd = sSVGDGassmannDet(d_obs=d_obs, sigma=sigma, device=device)

# 5) Run SVGD
n_iter  = 1000000
step_sz = 1e-3
bandw   = -1   # median trick
alpha   = 0.9

# If you want to track every iteration’s particles:
particle_history = svgd.update(
    x0=x0,
    n_iter=n_iter,
    stepsize=step_sz,
    bandwidth=bandw,
    alpha=alpha,
    debug=True,
    track_history=True,
)
print("particle_history.shape:", particle_history.shape)
# → (n_iter+1, num_particles, n_theta)

# 6) Discard burn‐in and flatten
burn_in = 1000
thin=10
chains = particle_history[burn_in:, :, :]  # shape = (n_iter+1 - burn_in, num_particles, n_theta)

chains_thinned = chains[::thin, :, :] 


samples = chains_thinned.reshape(-1, n_theta)      # shape = ((n_iter+1 - burn_in)*num_particles, n_theta)
print("Flattened samples shape:", samples.shape)

# 7) (Optional) Plot pairplot of θ‐samples

save_path = "src/example/marg/results/ssvgd_det.png"

m_true=torch.tensor([4, 7])

samples=np.vstack(chains)

np.save("src/example/samples/ssvgd_samples_det.npy",samples)

pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)