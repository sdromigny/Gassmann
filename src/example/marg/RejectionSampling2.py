import numpy as np
import torch
from scipy.special import logsumexp  # only used later on CPU
import math
from sklearn.neighbors import KernelDensity
import numpy as np
import os
import sys
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
import torch

from utilities.Histogram2d import pairplot



# ---- user inputs (tweak as needed) ----
m_samples_path = "/content/mcmc_samples_prob.npy"   # your file
M = 400            # nuisance draws per theta
batch_size = 256   # tune to fit your GPU memory (reduce if OOM)
sigma = 0.01
d_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
obs_dim = d_obs.size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- helper: sample nuisances once and move to GPU ----
def sample_nuis_parameters_numpy(n_samples=1):
    G_frame   = np.random.normal(8.5, 0.3, size=(n_samples,1)).astype(np.float32)
    porosity  = np.random.normal(0.37, 0.02, size=(n_samples,1)).astype(np.float32)
    rho_grain = np.random.normal(44.8, 0.8, size=(n_samples,1)).astype(np.float32)
    return np.hstack([G_frame, porosity, rho_grain])

nuisances_np = sample_nuis_parameters_numpy(M)           # shape (M,3) on CPU
nuisances_t = torch.from_numpy(nuisances_np).to(device)  # (M,3) on GPU

# put d_obs on GPU
d_obs_t = torch.from_numpy(d_obs).to(device).view(1, 1, obs_dim)  # (1,1,obs_dim) broadcastable

# ---- vectorized GPU simulator: returns sims shape (B, M, obs_dim) ----
# Replace internals if your simulator is different; this follows your earlier formula.
def simulator_batch_theta_n(theta_batch_t, nuisances_t):
    # theta_batch_t: (B, dim) torch float32
    # nuisances_t: (M, 3)
    # returns sims (B, M, obs_dim) and a boolean valid mask (B, M)
    # Here we assume the "theta uses first component" model from before:
    B = theta_batch_t.shape[0]
    theta0 = theta_batch_t[:, 0].view(B, 1)                 # (B,1)
    G = nuisances_t[:, 0].view(1, -1)                       # (1,M)
    por = nuisances_t[:, 1].view(1, -1)                     # (1,M)
    rho = nuisances_t[:, 2].view(1, -1)                     # (1,M)

    # Broadcast to (B, M)
    denominator = theta0 * (1.0 - por) + (por * rho)        # (B,M)
    valid = denominator > 0.0                               # (B,M) boolean

    # compute sims where valid
    # guard against divide-by-zero by clamping denominator slightly if needed
    # but we rely on `valid` mask to ignore invalid entries.
    denom_safe = torch.where(valid, denominator, torch.ones_like(denominator))
    sim_scalar = torch.sqrt(G / denom_safe)                 # shape (B,M) because G broadcasts
    # where invalid, sim_scalar value is meaningless; we'll mask later

    # Expand scalar into obs_dim channels if simulator is scalar
    sims = sim_scalar.unsqueeze(2).expand(-1, -1, obs_dim)  # (B,M,obs_dim)

    return sims, valid

# ---- load m_samples with mmap to avoid full-memory load if file is huge ----
m_samples = np.load(m_samples_path, mmap_mode="r")   # lazy-backed on disk
N = m_samples.shape[0]
print("Total m samples:", N, "dim:", m_samples.shape[1])

# allocate arrays (float32) to store per-sample results
vi_log_exp_loglike = np.empty(N, dtype=np.float32)
marginal_loglike = np.empty(N, dtype=np.float32)

# prior: vectorized version in numpy for speed (uniform [0,10]^D => 0 or -inf)
def log_prior_theta_vec(m_batch_np):
    # m_batch_np: (B, dim)
    inside = np.all((m_batch_np >= 0.0) & (m_batch_np <= 10.0), axis=1)
    out = np.full(m_batch_np.shape[0], -np.inf, dtype=np.float32)
    out[inside] = 0.0
    return out

# constants for likelihood
const_term = -0.5 * obs_dim * math.log(2.0 * math.pi * (sigma ** 2))
log_neg_inf = -1e30   # use a very negative number as -inf for numeric stability

# ---- main batched loop ----
start = 0
while start < N:
    end = min(start + batch_size, N)
    m_block_np = np.array(m_samples[start:end], dtype=np.float32)  # read from memmap (small chunk)
    B = m_block_np.shape[0]

    # copy to GPU
    theta_batch_t = torch.from_numpy(m_block_np).to(device)  # (B, dim)

    # get sims and valid mask for the whole block (B, M, obs_dim), (B, M)
    with torch.no_grad():
        sims_t, valid_t = simulator_batch_theta_n(theta_batch_t, nuisances_t)

        # compute squared distances: (B, M)
        diffs_t = sims_t - d_obs_t  # broadcast (1,1,obs_dim) -> (B,M,obs_dim)
        sq_norms_t = (diffs_t * diffs_t).sum(dim=2)  # (B,M)

        # compute per-nuisance log-likelihoods, set invalid ones to large negative
        logliks_t = const_term - 0.5 * (sq_norms_t / (sigma ** 2))  # (B,M)
        logliks_t = torch.where(valid_t, logliks_t, torch.tensor(log_neg_inf, device=device, dtype=logliks_t.dtype))

        # marginal: logsumexp over valid entries, normalize by number_valid (per theta)
        # compute valid counts (B,)
        valid_counts = valid_t.sum(dim=1)  # (B,) int
        valid_counts_cpu = valid_counts.cpu().numpy()

        # compute torch logsumexp
        marginal_logsumexp_t = torch.logsumexp(logliks_t, dim=1)  # (B,)

        # convert to numpy scalars: if valid_counts == 0 => mark large negative
        marginal_block = marginal_logsumexp_t.cpu().numpy()
        # subtract log(valid_counts) for averaging over valid draws
        # but be careful with zero valid_counts
        for j in range(B):
            k = int(valid_counts_cpu[j])
            if k == 0:
                marginal_loglike[start + j] = -np.inf
            else:
                marginal_loglike[start + j] = float(marginal_block[j]) - math.log(k)

        # VI-style expected log-likelihood: mean of logliks over valid entries
        # sum over M and divide by valid_counts (avoid division by zero)
        sum_logliks_t = logliks_t.sum(dim=1)  # (B,)
        sum_logliks_block = sum_logliks_t.cpu().numpy()
        for j in range(B):
            k = int(valid_counts_cpu[j])
            if k == 0:
                vi_log_exp_loglike[start + j] = -np.inf
            else:
                # sum_logliks_block[j] contains invalid entries as large negative -> included in sum
                # but since invalid entries are log_neg_inf they won't materially change sum if k>0.
                vi_log_exp_loglike[start + j] = float(sum_logliks_block[j]) / float(k)

    # free GPU memory quickly
    del theta_batch_t, sims_t, valid_t, diffs_t, sq_norms_t, logliks_t, marginal_logsumexp_t, sum_logliks_t
    torch.cuda.empty_cache()

    start = end

# ---- now compute weights on CPU (no q(m) used; treating samples as empirical) ----
log_prior_vals = log_prior_theta_vec(np.array(m_samples, dtype=np.float32))  # vectorized on memmap

# convert -inf to large negative so further ops remain numeric; but keep -inf for clarity
# form log-weights
logw_vi = vi_log_exp_loglike + log_prior_vals
logw_marg = marginal_loglike + log_prior_vals

# replace -inf with a very negative finite number for stability before logsumexp
logw_vi_safe = np.where(np.isfinite(logw_vi), logw_vi, -1e300)
logw_marg_safe = np.where(np.isfinite(logw_marg), logw_marg, -1e300)

# normalize
logw_vi_norm = logw_vi_safe - logsumexp(logw_vi_safe)
logw_marg_norm = logw_marg_safe - logsumexp(logw_marg_safe)

w_vi = np.exp(logw_vi_norm)
w_marg = np.exp(logw_marg_norm)

# ESS
ess_vi = 1.0 / np.sum(w_vi ** 2)
ess_marg = 1.0 / np.sum(w_marg ** 2)
print("ESS VI:", ess_vi, "ESS marg:", ess_marg)


# Resample with replacement according to the normalized weights
N_resample = N
idx_vi = np.random.choice(N, size=N_resample, replace=True, p=w_vi)
idx_marg = np.random.choice(N, size=N_resample, replace=True, p=w_marg)

resampled_vi = m_samples[idx_vi]
resampled_marg = m_samples[idx_marg]





m_true = np.array([4, 7])  # if you want to mark the true parameters
pairplot(m_samples, m_true=m_true,
                                 save_path="/content/mcmc.png")
pairplot(resampled_vi, m_true=m_true,
                                 save_path="/content/resampled_vi.png")
pairplot(resampled_marg, m_true=m_true,
                                 save_path="/content/resampled_marg.png")
