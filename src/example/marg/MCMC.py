
import numpy as np
import torch
import math
import time
import os
import sys
import time

current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

from tqdm import trange
from utilities.Gassmann import simulator_prob, simulator_det, sample_nuis_parameters_numpy
from utilities.Histogram2d import pairplot

from utilities.MCMCFunc import *



import time
start_time = time.perf_counter()

# Set observed data and uncertainty
d_obs = np.array([0.64704126, 0.61732611])
sigma = 0.01

# Set log likelihood definition
model = ExampleModel(d_obs, sigma)

# Set inversion parameters
nchains = 10
xs = [np.array([1.0, 5.0])] * nchains


# Run the McMC sampler
ctrl = PseudoMarginalMCMCController(model, nchains, xs)
ctrl.set_max_iterations(1000000)
ctrl.set_prop_scale(0.05)
ctrl.set_report_interval(5000)

chains = ctrl.run(verbose=True)

print("chains shape:", chains.shape)


burn_in = 100000   
thin = 10         
save_path = "./src/example/marg/results/mcmc_prob_pmh_time.png"
out_npy = "./src/example/samples/marg/mcmc_samples_prob_pmh_time.npy"
m_true = torch.tensor([4.0, 7.0])


chains = np.asarray(chains)
if chains.ndim != 3:
    raise ValueError(f"Expected chains array of shape (n_chains, n_iters, D). Got shape {chains.shape}")

n_chains, n_iters, D = chains.shape
print(f"Raw chains shape: n_chains={n_chains}, n_iters={n_iters}, D={D}")


if burn_in >= n_iters:
    suggested = max(0, int(0.1 * n_iters))
    warnings.warn(
        f"Requested burn_in={burn_in} >= n_iters={n_iters}. "
        f"Setting burn_in to {suggested} (10% of chain length).",
        UserWarning
    )
    burn_in = suggested

post_chains = chains[:, burn_in::thin, :]
n_post = post_chains.shape[1]
print(f"Using burn_in={burn_in}, thin={thin} -> post samples per chain = {n_post}")

# Save the samples and plot
samples = post_chains.reshape(-1, D)   
print("samples shape (after vstack):", samples.shape)



np.save(out_npy, samples)
print(f"Saved samples to {out_npy}")

if D >= 2:
    pairplot(samples[:, :2], m_true.detach().numpy(), fontsize=15, save_path=save_path)
else:
    raise ValueError("Samples have D < 2, cannot call pairplot for 2D.")

end_time = time.perf_counter()
print(f"Total runtime: {end_time - start_time:.2f} seconds")
