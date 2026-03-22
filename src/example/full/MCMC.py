import torch
import os
import sys
# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import pints
import matplotlib.pyplot as plt
from utilities.MCMCFunc import FullMCMC
from utilities.Gassmann import simulator_prob, simulator_det, sample_nuis_parameters_numpy
from utilities.PlotHighD import plot_5d_corner
import time
start_time = time.perf_counter()

# Initialise the chains
nchains = 10  # Total number of chains

# Define observed data
x_obs = np.array([0.64704126, 0.61732611])

# Sample initial values from priors
xs = []
for _ in range(nchains):
    main_params = np.random.uniform(0, 10, size=len(x_obs))
    latent_params = np.array([
        np.random.normal(8.5, 0.3),  # G_frame
        np.random.normal(0.37, 0.02),  # Porosity
        np.random.normal(44.8, 0.8)   # Rho_grain
    ])
    xs.append(np.concatenate([main_params, latent_params]))
xs = np.array(xs)

# Set uncertainty
sigma = 0.01


# Initialize the class
model = FullMCMC(x_obs, sigma)

# Initialize MCMC controller
mcmc = pints.MCMCController(model, nchains, xs, method=pints.MetropolisRandomWalkMCMC)

# Add stopping criterion
mcmc.set_max_iterations(5000000)


# Run the MCMC sampling
chains = mcmc.run()

# Define the burn-in period
burn_in = 500000  
thin=10
# Discard the first `burn_in` samples for all chains
chains = chains[:, burn_in:, :]  # Keep samples after burn-in

chains_thinned = chains[:, ::thin, :] 

save_path = "./src/example/full/results/mcmc_time.png"


# Save the samples and plot
samples=np.vstack(chains_thinned)

np.save("./src/example/samples/full/mcmc_samples_time.npy",samples)

print(samples.shape)

plot_5d_corner(samples, save_path=save_path)


end_time = time.perf_counter()

print(f"Total runtime: {end_time - start_time:.2f} seconds")
