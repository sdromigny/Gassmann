from sklearn.neighbors import KernelDensity
import numpy as np
import torch
import os
import sys
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
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
device="cpu"

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
def log_marginal_likelihood_mc(theta_batch_np, M=5000):

    # Convert to numpy array first
    arr = np.asarray(theta_batch_np)
    # If 1D, reshape to (1, D)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    elif arr.ndim != 2:
        raise ValueError(f"theta_batch_np must be 1D or 2D array, got shape {arr.shape}")

    theta_batch = torch.tensor(arr, dtype=torch.float32, device=device)  # (B, D)
    B, D = theta_batch.shape

    # Sample nuisances once per batch: shape (M, 3)
    n_np = sample_nuis_parameters_numpy(M)  # (M, 3)
    n_t = torch.tensor(n_np, dtype=torch.float32, device=device)  # (M, 3)

    # Prepare batched inputs for simulator:
    # Expand θ to (B, M, D) and n to (B, M, 3), then flatten
    theta_rep = theta_batch.unsqueeze(1).expand(-1, M, -1)  # (B, M, D)
    n_rep     = n_t.unsqueeze(0).expand(B, M, -1)          # (B, M, 3)

    theta_flat = theta_rep.reshape(-1, D)  # (B*M, D)
    n_flat     = n_rep.reshape(-1, 3)      # (B*M, 3)

    # Run simulator in no_grad mode (unless you need gradients). Make sure simulator_prob_n returns a torch tensor on device.
    with torch.no_grad():
        # d_pred_flat: shape (B*M, dim_obs)
        d_pred_flat = simulator_prob_n(theta_flat, n_flat)

    # Compute log-likelihood for each pair:
    # p(d_obs | θ, n) ~ Normal(d_pred, sigma^2 I)
    d_obs_t = torch.tensor(d_obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, dim_obs)
    # Broadcast to (B*M, dim_obs):
    d_obs_rep = d_obs_t.expand(d_pred_flat.shape[0], -1)

    # squared residuals:
    resid2 = ((d_obs_rep - d_pred_flat) / sigma)**2  # (B*M, dim_obs)
    sum_resid2 = torch.sum(resid2, dim=1)            # (B*M,)

    dim_obs = d_obs_t.shape[-1]
    # Include normalization constant if desired:
    #    log p = -0.5 * (dim_obs * log(2π σ^2) + sum_resid2)
    # If this constant is independent of θ, you may omit it in optimization but for a correct log marginal include it.
    const_term = -0.5 * (dim_obs * np.log(2 * np.pi * sigma**2))
    ll_flat = const_term - 0.5 * sum_resid2  # (B*M,)

    # Reshape back to (B, M)
    ll_mat = ll_flat.reshape(B, M)  # (B, M)

    # Compute log-sum-exp along axis=1: log( (1/M) ∑ e^{ll_i} ) = logsumexp(ll_mat, dim=1) - log(M)
    logsumexp_per_theta = torch.logsumexp(ll_mat, dim=1)  # (B,)
    log_M = float(np.log(M))
    log_marginal = logsumexp_per_theta - log_M            # (B,)

    return log_marginal.cpu().numpy()  # shape (B,)

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
        ell      = log_marginal_likelihood_mc(theta, M=M)
        lp_theta = log_prior_theta(theta)
        scores[i] = ell + lp_theta - log_q[i]

    s_thresh  = scores.mean()
    keep_mask = scores >= s_thresh
    return m_samples[keep_mask], scores, s_thresh

if __name__ == "__main__":
    # load your MCMC draws
    X = np.load("/home/users/scro4690/Documents/GenInv/Gassmann/src/example/samples/marg/mcmc_samples_prob.npy")
    filtered_X, scores, threshold = rejection_filter(X[:,:], kde_bw=1.5, M=10000)
    print(f"Kept {filtered_X.shape[0]} / {X.shape[0]} samples (threshold={threshold:.3f})")
    m_true=torch.tensor([4, 7])
    np.save("/home/users/scro4690/Documents/GenInv/Gassmann/src/example/samples/marg/control_mc.npy", filtered_X)

    save_path = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/marg/results/mcmc_control.png"
    pairplot(filtered_X, m_true.detach().cpu().numpy(), fontsize=15, save_path=save_path)




