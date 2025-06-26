from sklearn.neighbors import KernelDensity
import numpy as np
import os
import sys
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
import torch
from utilities.Gassmann import simulator_prob_n, simulator_det, sample_nuis_parameters_numpy
from utilities.Histogram2d import pairplot
from sklearn.neighbors import KernelDensity
import numpy as np
import torch
from utilities.Gassmann import simulator_prob, sample_nuis_parameters_numpy
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from utilities.Gassmann import simulator_prob_n  # make sure this takes (theta_torch, n_array)
from utilities.Gassmann import sample_nuis_parameters_numpy

# -- observed data and noise scale
d_obs = np.array([0.64704126, 0.61732611])
sigma = 0.01

# --- prior on the main parameters θ
def log_prior_theta(theta):
    # uniform over [0,10]^D
    if np.any((theta < 0) | (theta > 10)):
        return -np.inf
    return 0.0

# --- prior on the nuisance parameters n
def log_prior_n(n):
    # Gaussian priors on the three nuisance dims
    lp = -0.5 * ((n[0] - 8.5) / 0.3)**2
    lp += -0.5 * ((n[1] - 0.37) / 0.02)**2
    lp += -0.5 * ((n[2] - 44.8) / 0.8)**2
    return lp

# --- fix sample_nuis_parameters_numpy so it actually returns an array of shape (n_samples,3)
def sample_nuis_parameters_numpy(n_samples=1):
    G_frame   = np.random.normal(8.5, 0.3, size=(n_samples,1))
    porosity  = np.random.normal(0.37, 0.02, size=(n_samples,1))
    rho_grain = np.random.normal(44.8, 0.8, size=(n_samples,1))
    return np.hstack([G_frame, porosity, rho_grain])

# --- Monte Carlo estimate E_{p(n)}[ log p(d_obs | θ, n) ]
def E_log_likelihood_vec(theta, M=2000):
    """
    Vectorized: sample M nuisances at once, then do one batched call to simulator_prob_n.
    """
    # theta: 1D array of shape (D,)
    theta_t = torch.tensor(theta, dtype=torch.float32).unsqueeze(0)     # (1, D)
    
    # sample M nuisances in one shot
    n = sample_nuis_parameters_numpy(M)                                 # (M, 3)
    n_t = torch.tensor(n, dtype=torch.float32)                         # (M, 3)
    
    # repeat theta to shape (M, D)
    theta_rep = theta_t.expand(M, -1)                                   # (M, D)
    
    # one batched simulation
    d_pred = simulator_prob_n(theta_rep, n_t)                          # (M, 2)
    
    # compute all likelihoods at once
    d_obs_t = torch.tensor(d_obs, dtype=torch.float32).unsqueeze(0)     # (1,2)
    residual = (d_obs_t - d_pred) / sigma                               # (M,2)
    ll_all = -0.5 * torch.sum(residual**2, dim=1)                       # (M,)
    
    return ll_all.mean().item()


def approximate_kde_2d(samples, bandwidth, grid_size=512, eps=1e-12):
    """
    Approximate 2D KDE via histogram binning and FFT-based convolution.
    
    Parameters:
    - samples: np.ndarray of shape (N, 2)
    - bandwidth: scalar bandwidth for Gaussian kernel
    - grid_size: number of bins along each axis for the histogram grid
    - eps: small constant to avoid log(0)
    
    Returns:
    - log_density: np.ndarray of shape (N,), approximate log density at each sample
    """
    # Determine data bounds with a margin
    x = samples[:, 0]
    y = samples[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    # Add a small margin so edge points are properly handled
    margin_x = bandwidth * 3
    margin_y = bandwidth * 3
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y

    # Histogram bin edges
    x_edges = np.linspace(x_min, x_max, grid_size + 1)
    y_edges = np.linspace(y_min, y_max, grid_size + 1)

    # Compute 2D histogram
    hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    hist = hist.astype(np.float64)

    # Grid cell sizes
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size

    # Prepare Gaussian kernel on the grid
    # Kernel grid coordinates: centered at zero
    gx = np.fft.fftfreq(grid_size, d=dx)
    gy = np.fft.fftfreq(grid_size, d=dy)
    # Create 2D frequency grid
    kx, ky = np.meshgrid(gx, gy, indexing='xy')
    # Fourier transform of Gaussian: exp(-2π^2 * bandwidth^2 * (kx^2 + ky^2))
    gaussian_ft = np.exp(-2 * (np.pi**2) * (bandwidth**2) * (kx**2 + ky**2))

    # FFT-based convolution: density_grid = ifft( fft(hist) * gaussian_ft )
    hist_ft = np.fft.fft2(hist)
    density_grid = np.fft.ifft2(hist_ft * gaussian_ft).real

    # Normalize: KDE density = density_grid / (N * dx * dy)
    N = samples.shape[0]
    density_grid /= (N * dx * dy)

    # Map each sample to grid index
    # Compute bin indices for x and y (clamp to [0, grid_size-1])
    ix = np.clip(((x - x_min) / dx).astype(int), 0, grid_size - 1)
    iy = np.clip(((y - y_min) / dy).astype(int), 0, grid_size - 1)

    # Retrieve density values
    density_vals = density_grid[ix, iy]

    # Convert to log density, adding eps to avoid log(0)
    log_density = np.log(density_vals + eps)
    return log_density


from tqdm import tqdm

def rejection_filter(m_samples, kde_bw=0.5, M=2000):
    N = len(m_samples)
    kde   = KernelDensity(bandwidth=kde_bw, kernel="gaussian").fit(m_samples)
    print("kde fitted")
    log_q = approximate_kde_2d(m_samples, kde_bw)
    print("compute log q")
    scores = np.empty(N)

    for i, theta in enumerate(tqdm(m_samples, desc="Scoring samples")):
        ell      = E_log_likelihood_vec(theta, M=M)
        lp_theta = log_prior_theta(theta)
        scores[i] = ell + lp_theta - log_q[i]

    s_thresh  = scores.mean()
    keep_mask = scores >= s_thresh
    return m_samples[keep_mask], scores, s_thresh

if __name__ == "__main__":
    # load your MCMC draws
    X = np.load("/home/users/scro4690/Documents/GenInv/Gassmann/src/example/samples/marg/mcmc_samples_prob.npy")
    filtered_X, scores, threshold = rejection_filter(X[:,:], kde_bw=0.3, M=10)
    print(f"Kept {filtered_X.shape[0]} / {X.shape[0]} samples (threshold={threshold:.3f})")
    m_true=torch.tensor([4, 7])
    np.save("/home/users/scro4690/Documents/GenInv/Gassmann/src/example/samples/marg/filtered_mc.npy", filtered_X)
    save_path = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/marg/results/mcmc_reject.png"
    pairplot(filtered_X, m_true.detach().numpy(), fontsize=15, save_path=save_path)




