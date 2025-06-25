import numpy as np
import torch


def log_normal_pdf(x, mu, sigma):
    """
    Compute log of the normal PDF at x for N(mu, sigma^2).
    """
    return -0.5 * ((x - mu) / sigma)**2 \
           - np.log(sigma) \
           - 0.5 * np.log(2 * np.pi)


def sample_and_log_gaussians(samples, params=None, device=None):
    """
    Draw from a product of independent Gaussians and compute log-pdfs in torch.

    Args:
      n_samples (int): number of samples to draw
      params (list of (mu, sigma) tuples):  
          defaults to [(8.5, 0.3), (0.37, 0.02), (44.8, 0.8)]
      device (torch.device or str): where to allocate tensors (e.g. 'cuda:0')

    Returns:
      samples  : Tensor of shape (n_samples, 3)
      log_pdfs : Tensor of shape (n_samples, 3)
      log_joint: Tensor of shape (n_samples,)
    """
    if params is None:
        params = [(8.5, 0.3), (0.37, 0.02), (44.8, 0.8)]
    device = torch.device(device) if device is not None else torch.device('cpu')

    # Create torch Normal distributions
    dists = [torch.distributions.Normal(loc=mu, scale=sigma) 
             for mu, sigma in params]


    # Compute individual log-pdfs: same shape as samples
    log_pdfs = torch.stack([dist.log_prob(samples[:, i])
                             for i, dist in enumerate(dists)], dim=1)  # (n_samples, 3)

    # Joint log-prob is sum across the 3 Gaussians
    log_joint = log_pdfs.sum(dim=1)  # (n_samples,)

    return log_joint


def sample_nuis_parameters_numpy(n_samples):
    G_frame = np.random.normal(8.5, 0.3, n_samples)     # G_frame
    porosity = np.random.normal(0.37, 0.02, n_samples)  # Porosity
    rho_grain = np.random.normal(44.8, 0.8, n_samples)  # Rho_grain
    return np.stack([G_frame, porosity, rho_grain], axis=1)

def sample_nuis_parameters_cuda(n_samples, device='cpu'):
    return torch.stack([
        torch.normal(8.5, 0.3, (n_samples,), device=device),
        torch.normal(0.37, 0.02, (n_samples,), device=device),
        torch.normal(44.8, 0.8, (n_samples,), device=device)
    ], dim=1)

def simulator_prob(theta):
    max_attempts = 1_000_000
    is_torch = torch.is_tensor(theta)
    
    if is_torch:
        batch_size = theta.shape[0] if theta.ndim > 1 else 1
        device = theta.device
    else:
        batch_size = theta.shape[0] if theta.ndim > 1 else 1

    for attempt in range(max_attempts):
        if is_torch:
            m = sample_nuis_parameters_cuda(batch_size, device=device)
            theta_exp = theta if theta.ndim > 1 else theta.unsqueeze(0)
            denominator = (theta_exp * (1 - m[:, 1:2])) + (m[:, 1:2] * m[:, 2:3])
            if (denominator > 0).all().item():
                sim_data = torch.sqrt(m[:, :1] / denominator)
                return sim_data.view(batch_size, 2),m
        else:
            m = sample_nuis_parameters_numpy(batch_size)
            theta_exp = theta if theta.ndim > 1 else theta[np.newaxis, :]
            denominator = (theta_exp * (1 - m[:, 1:2])) + (m[:, 1:2] * m[:, 2:3])
            if (denominator > 0).all():
                sim_data = np.sqrt(m[:, :1] / denominator)
                return sim_data.reshape(batch_size, 2),m

    print("Warning: All attempts to find a valid denominator failed.")
    if is_torch:
        return torch.full((batch_size, 2), float('nan'), device=device)
    else:
        return np.full((batch_size, 2), np.nan)



##################################################################################
def simulator_det(x):
    """
    Forward model with resampling of parameters until all denominator values > 0.

    Args:
        x (torch.Tensor): Input tensor for the model.
        max_attempts (int): Maximum attempts to resample valid parameters (unused here but kept for compatibility).

    Returns:
        torch.Tensor: Synthetic data, with NaN if denominator is invalid.
    """
    # Initialize synthetic values
    synthetic_value = torch.zeros_like(x)  # Placeholder for synthetic values

    # Fixed deterministic parameters
    G_frame_values = torch.tensor([7.994777040351033, 7.997586766477148], device=x.device)
    poro = 0.3700000000000001
    rho_grain = 44.8

    # Iterate over each element in x
    for i in range(x.size(0)):
        # Use modulo to wrap around G_frame_values if x has more elements
        G_frame = G_frame_values[i % len(G_frame_values)]

        # Compute the denominator
        denominator = (x[i] * (1 - poro)) + (poro * rho_grain)


        synthetic_value[i] = torch.sqrt(G_frame / denominator)

    return synthetic_value



def simulator_prob_n(theta,n):
    max_attempts = 1_000_000
    is_torch = torch.is_tensor(theta)
    
    if is_torch:
        batch_size = theta.shape[0] if theta.ndim > 1 else 1
        device = theta.device
    else:
        batch_size = theta.shape[0] if theta.ndim > 1 else 1

    for attempt in range(max_attempts):
        if is_torch:
            theta_exp = theta if theta.ndim > 1 else theta.unsqueeze(0)
            denominator = (theta_exp * (1 - n[:, 1:2])) + (n[:, 1:2] * n[:, 2:3])
            if (denominator > 0).all().item():
                sim_data = torch.sqrt(n[:, :1] / denominator)
                return sim_data.view(batch_size, 2)
        else:
            theta_exp = theta if theta.ndim > 1 else theta[np.newaxis, :]
            denominator = (theta_exp * (1 - n[:, 1:2])) + (n[:, 1:2] * n[:, 2:3])
            if (denominator > 0).all():
                sim_data = np.sqrt(n[:, :1] / denominator)
                return sim_data.reshape(batch_size, 2)

    print("Warning: All attempts to find a valid denominator failed.")
    if is_torch:
        return torch.full((batch_size, 2), float('nan'), device=device)
    else:
        return np.full((batch_size, 2), np.nan)


def simulator_det_cuda(x):
    # x is (N, n_theta), but you only use x[:,0] in the current code?
    # Let’s assume x is shape (N,) for simplicity; generalize if needed.

    # 1) choose G_frame per sample
    #    If it truly alternates between two constants:
    G0, G1 = 7.994777040351033, 7.997586766477148
    # create a mask [True, False, True, False, ...] of length N
    idx = torch.arange(x.shape[0], device=x.device)
    use_G0 = (idx % 2 == 0)
    G_frame = torch.where(use_G0, G0, G1).unsqueeze(-1)  # shape (N,1) or (N,)

    poro      = 0.37
    rho_grain = 44.8

    denominator = x * (1 - poro) + poro * rho_grain  # (N,)
    return torch.sqrt(G_frame / denominator)         # (N,)



import torch

def simulator_full(theta):
    """
    Simulate data for a batch of parameter vectors, each of length 8:
      [ main0, main1, G_frame0, G_frame1, poro0, poro1, rho_grain0, rho_grain1 ]

    Returns:
      sim_data: torch.Tensor of shape (batch_size, 2)
                sim_data[:, 0] corresponds to the simulation using the first set,
                sim_data[:, 1] using the second set.

    If denominator <= 0 for any channel, that entry is set to NaN (or you can choose another handling).
    """
    # Ensure theta is a torch tensor
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.float32)

    # Handle single sample: expand to batch dimension
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # shape (1, 8)

    # Now theta.ndim == 2
    batch_size, D = theta.shape
    if D != 8:
        raise ValueError(f"Expected theta with 8 elements per sample, but got shape {theta.shape}")

    # Extract slices
    # main parameters:
    main_params = theta[:, 0:2]     # shape (batch_size, 2)
    # G_frame:
    G_frame = theta[:, 2:4]         # shape (batch_size, 2)
    # porosity:
    porosity = theta[:, 4:6]        # shape (batch_size, 2)
    # rho_grain:
    rho_grain = theta[:, 6:8]       # shape (batch_size, 2)

    # Compute denominator: shape (batch_size, 2)
    denominator = main_params * (1.0 - porosity) + porosity * rho_grain

    # Check invalid entries
    invalid_mask = denominator <= 0  # shape (batch_size, 2), True where invalid

    # Compute safe denominator for sqrt: add small epsilon to avoid zero division,
    # but we'll mask invalid afterward.
    eps = 1e-12
    safe_denominator = denominator.clamp_min(eps)

    # Compute simulation: shape (batch_size, 2)
    sim = torch.sqrt(G_frame / safe_denominator)

    # Mark invalid results as NaN (optional)
    # If you prefer another handling (e.g., large negative log-likelihood later),
    # you can leave sim values but pass invalid_mask upward.
    sim = torch.where(invalid_mask, torch.tensor(float('nan'), device=theta.device), sim)

    # If you prefer to raise an error when invalid occurs:
    # if invalid_mask.any():
    #     raise RuntimeError(f"Invalid denominator <= 0 for some samples/channels in simulator_full.")

    return sim  # shape (batch_size, 2)

