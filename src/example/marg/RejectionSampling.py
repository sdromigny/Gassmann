import numpy as np
import torch
from scipy.special import logsumexp  
from sklearn.neighbors import KernelDensity
import numpy as np
import os
import sys
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
import torch
from utilities.Gassmann import *
from utilities.Histogram2d import pairplot
import math

m_samples_path = "./src/example/samples/marg/mcmc_samples_prob_pmh.npy"   
M = 400            
batch_size = 256   
sigma = 0.01
d_obs = np.array([0.64704126, 0.61732611], dtype=np.float32)
obs_dim = d_obs.size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




nuisances_np = sample_nuis_parameters_numpy1d(M)           
nuisances_t = torch.from_numpy(nuisances_np).to(device)  


d_obs_t = torch.from_numpy(d_obs).to(device).view(1, 1, obs_dim)  



m_samples = np.load(m_samples_path)   
N = m_samples.shape[0]
print("Total m samples:", N, "dim:", m_samples.shape[1])


vi_log_exp_loglike = np.empty(N, dtype=np.float32)
marginal_loglike = np.empty(N, dtype=np.float32)


def log_prior_theta_vec(m_batch_np):

    inside = np.all((m_batch_np >= 0.0) & (m_batch_np <= 10.0), axis=1)
    out = np.full(m_batch_np.shape[0], -np.inf, dtype=np.float32)
    out[inside] = 0.0
    return out


const_term = -0.5 * obs_dim * math.log(2.0 * math.pi * (sigma ** 2))
log_neg_inf = -1e30   


start = 0
while start < N:
    end = min(start + batch_size, N)
    m_block_np = np.array(m_samples[start:end], dtype=np.float32)  
    B = m_block_np.shape[0]

    theta_batch_t = torch.from_numpy(m_block_np).to(device)  


    with torch.no_grad():
        sims_t, valid_t = simulator_batch_theta_n(theta_batch_t, nuisances_t, obs_dim)

 
        diffs_t = sims_t - d_obs_t  
        sq_norms_t = (diffs_t * diffs_t).sum(dim=2) 


        logliks_t = const_term - 0.5 * (sq_norms_t / (sigma ** 2))  
        logliks_t = torch.where(valid_t, logliks_t, torch.tensor(log_neg_inf, device=device, dtype=logliks_t.dtype))

        
        valid_counts = valid_t.sum(dim=1)  
        valid_counts_cpu = valid_counts.cpu().numpy()


        marginal_logsumexp_t = torch.logsumexp(logliks_t, dim=1)  

      
        marginal_block = marginal_logsumexp_t.cpu().numpy()
        
        for j in range(B):
            k = int(valid_counts_cpu[j])
            if k == 0:
                marginal_loglike[start + j] = -np.inf
            else:
                marginal_loglike[start + j] = float(marginal_block[j]) - math.log(k)

        
        sum_logliks_t = logliks_t.sum(dim=1)  
        sum_logliks_block = sum_logliks_t.cpu().numpy()
        for j in range(B):
            k = int(valid_counts_cpu[j])
            if k == 0:
                vi_log_exp_loglike[start + j] = -np.inf
            else:
                
                vi_log_exp_loglike[start + j] = float(sum_logliks_block[j]) / float(k)

   
    del theta_batch_t, sims_t, valid_t, diffs_t, sq_norms_t, logliks_t, marginal_logsumexp_t, sum_logliks_t
    torch.cuda.empty_cache()

    start = end


log_prior_vals = log_prior_theta_vec(np.array(m_samples, dtype=np.float32)) 


logw_vi = vi_log_exp_loglike + log_prior_vals
logw_marg = marginal_loglike + log_prior_vals


logw_vi_safe = np.where(np.isfinite(logw_vi), logw_vi, -1e300)
logw_marg_safe = np.where(np.isfinite(logw_marg), logw_marg, -1e300)


logw_vi_norm = logw_vi_safe - logsumexp(logw_vi_safe)
logw_marg_norm = logw_marg_safe - logsumexp(logw_marg_safe)

w_vi = np.exp(logw_vi_norm)
w_marg = np.exp(logw_marg_norm)

# ESS
ess_vi = 1.0 / np.sum(w_vi ** 2)
ess_marg = 1.0 / np.sum(w_marg ** 2)
print("ESS VI:", ess_vi, "ESS marg:", ess_marg)

N_resample = N
idx_vi = np.random.choice(N, size=N_resample, replace=True, p=w_vi)
idx_marg = np.random.choice(N, size=N_resample, replace=True, p=w_marg)

resampled_vi = m_samples[idx_vi]
resampled_marg = m_samples[idx_marg]





m_true = np.array([4, 7]) 
pairplot(m_samples, m_true=m_true,
                                 save_path="src/example/marg/results/mcmc_time.png")
pairplot(resampled_vi, m_true=m_true,
                                 save_path="src/example/marg/results/resampled_vi_time.png")
pairplot(resampled_marg, m_true=m_true,
                                 save_path="src/example/marg/results/resampled_marg_time.png")
