
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

from utilities.MCMCFunc import MCSamplingGassmannProb, MCSamplingGassmannDet, MCSamplingGassmannIndepProb
from utilities.Gassmann import sample_nuis_parameters_numpy
from utilities.Histogram2d import pairplot


# Initialise the chains
nchains = 10 # total number of chains
xs = [[1, 5]] * nchains # initial positions for all chains


# Properties of the observed data (assumed error: sigma, m: prior, x_obs: observed data values)
sigma=0.01
m=sample_nuis_parameters_numpy(1)
x_obs = np.array([0.64704126, 0.61732611])




# Initialize the class
model = MCSamplingGassmannIndepProb(x_obs, sigma) 

# Initialize MCMC controller
mcmc = pints.MCMCController(model, nchains, xs, method=pints.MetropolisRandomWalkMCMC)

# Add stopping criterion
mcmc.set_max_iterations(1000000)

# Run the MCMC sampling
chains = mcmc.run()

# Define the burn-in period
burn_in = 1000# Adjust this value based on your needs

# Discard the first `burn_in` samples for all chains
chains = chains[:, burn_in:, :]  # Keep samples after burn-in



save_path = "src/example/marg/results/mcmc_prob_indep.png"

m_true=torch.tensor([4, 7])

samples=np.vstack(chains)

np.save("src/example/samples/mcmc_samples_prob_indep.npy",samples)

print(samples.shape)

pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)

########################################################################################################

# # Initialise the chains
nchains = 10 # total number of chains
xs = [[1, 5]] * nchains # initial positions for all chains


# Properties of the observed data (assumed error: sigma, m: prior, x_obs: observed data values)
sigma=0.01

x_obs = np.array([0.64704126, 0.61732611])

# Initialize the class
model = MCSamplingGassmannDet(x_obs, sigma) 

# Initialize MCMC controller
mcmc = pints.MCMCController(model, nchains, xs, method=pints.MetropolisRandomWalkMCMC)

# Add stopping criterion
mcmc.set_max_iterations(1000000)

# Run the MCMC sampling
chains = mcmc.run()

# Define the burn-in period
burn_in = 1000  # Adjust this value based on your needs


# Discard the first `burn_in` samples for all chains
chains = chains[:, burn_in:, :]  # Keep samples after burn-in

save_path = "src/example/marg/results/mcmc_det.png"

m_true=torch.tensor([4, 7])

samples=np.vstack(chains)

np.save("src/example/samples/mcmc_samples_det.npy",samples)

print(samples.shape)

pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)


