import time
import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import torch
from utilities.SVGDFunc import sSVGDGassmannProb, sSVGDGassmannDet  
from utilities.Histogram2d import pairplot

####################################################################################################################################
### Probabilistic version ###
start_time = time.perf_counter()

# Set observed data and uncertainty
d_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
sigma = 0.01

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set inversion parameters
num_particles = 100
n_theta = 2  
x0 = np.random.uniform(0.0, 10.0, size=(num_particles, n_theta)).astype(np.float32)


svgd = sSVGDGassmannProb(d_obs=d_obs, sigma=sigma, device=device)

n_iter  = 10000
step_sz = 1e-3
bandw   = -1   
alpha   = 0.9

# Call the sampler
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



burn_in = 1000
chains = particle_history[burn_in:, :, :]  
samples = chains.reshape(-1, n_theta)     
print("Flattened samples shape:", samples.shape)

# Save the samples and plot
save_path = "./src/example/marg/results/ssvgd_prob_time.png"

m_true=torch.tensor([4, 7])

samples=np.vstack(chains)

np.save("./src/example/samples/marg/ssvgd_samples_prob_time.npy",samples)

pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)

end_time = time.perf_counter()

print(f"Total runtime: {end_time - start_time:.2f} seconds")


##################################################################################################################
### Deterministic version ###

start_time = time.perf_counter()

# Set observed data and uncertainty
d_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
sigma = 0.01

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Set inversion parameters
num_particles = 100

n_theta = 2  
x0 = np.random.uniform(0.0, 10.0, size=(num_particles, n_theta)).astype(np.float32)


svgd = sSVGDGassmannDet(d_obs=d_obs, sigma=sigma, device=device)


n_iter  = 10000

step_sz = 1e-3
bandw   = -1   
alpha   = 0.9

# Call sampler
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

burn_in = 1000

thin=10

chains = particle_history[burn_in:, :, :]  

chains_thinned = chains[::thin, :, :] 


samples = chains_thinned.reshape(-1, n_theta)      
print("Flattened samples shape:", samples.shape)



save_path = "./src/example/marg/results/ssvgd_det_time.png"

m_true=torch.tensor([4, 7])

samples=np.vstack(chains)


# Save the samples and plot

np.save("./src/example/samples/marg/ssvgd_samples_det_time.npy",samples)

pairplot(samples, m_true.detach().numpy(), fontsize=15, save_path=save_path)

end_time = time.perf_counter()

print(f"Total runtime: {end_time - start_time:.2f} seconds")
