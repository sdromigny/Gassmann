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




import torch

def simulator_prob_indep(theta, max_attempts=1_000_000):
    """
    Simulator with independent nuisance draws for each θ-dimension.
    
    theta: tensor shape (batch_size, 2) or (2,)
    Returns:
      sim_data: tensor shape (batch_size, 2)
      m:        tensor shape (batch_size, 2, 3) of nuisance draws
    """
    is_torch = torch.is_tensor(theta)
    if is_torch:
        device = theta.device
        theta_exp = theta if theta.ndim > 1 else theta.unsqueeze(0)
    else:
        theta_exp = torch.tensor(theta, dtype=torch.float32)
        device = torch.device("cpu")
        theta_exp = theta_exp.unsqueeze(0) if theta_exp.ndim == 1 else theta_exp

    batch_size = theta_exp.shape[0]
    if theta_exp.shape[1] != 2:
        raise ValueError(f"Expected theta of shape (batch,2), got {theta_exp.shape}")

    for _ in range(max_attempts):
        # 1) draw 2 * batch_size nuisance triples
        m_all = sample_nuis_parameters_cuda(batch_size * 2, device=device) \
                if is_torch else sample_nuis_parameters_numpy(batch_size * 2)
        # shape (batch_size*2, 3)
        
        # 2) reshape to (batch_size, 2, 3)
        m = m_all.view(batch_size, 2, 3) if is_torch else m_all.reshape(batch_size, 2, 3)
        
        # 3) split by channel
        theta_ch0 = theta_exp[:, 0]  # (batch,)
        theta_ch1 = theta_exp[:, 1]  # (batch,)

        # nuisance for each channel
        num0, poro0, grain0 = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
        num1, poro1, grain1 = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]

        denom0 = theta_ch0 * (1 - poro0) + poro0 * grain0
        denom1 = theta_ch1 * (1 - poro1) + poro1 * grain1

        # check validity
        valid0 = denom0 > 0
        valid1 = denom1 > 0

        # only accept if *all* batch entries have both channels valid
        if (valid0 & valid1).all().item():
            sim0 = torch.sqrt(num0 / denom0) if is_torch else np.sqrt(num0 / denom0)
            sim1 = torch.sqrt(num1 / denom1) if is_torch else np.sqrt(num1 / denom1)

            sim = torch.stack([sim0, sim1], dim=1) if is_torch else np.stack([sim0, sim1], axis=1)
            return sim, m

    # fallback if no valid draws
    print("Warning: All attempts to find a valid denominator failed.")
    if is_torch:
        return torch.full((batch_size, 2), float('nan'), device=device), None
    else:
        return np.full((batch_size, 2), np.nan), None



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
def simulator_full(theta, max_attempts=100000):
    """
    Simulate data for a batch of parameter vectors, each of length 8:
      [ main0, main1, G_frame0, G_frame1, poro0, poro1, rho_grain0, rho_grain1 ]
    For any channel where denominator <= 0, resample parameters from their priors until valid.

    Returns:
      sim: torch.Tensor of shape (batch_size, 2)
    """
    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, dtype=torch.float32)

    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # shape (1, 8)

    batch_size, D = theta.shape
    if D != 8:
        raise ValueError(f"Expected theta with 8 elements per sample, but got shape {theta.shape}")

    # Extract and clone slices so we can resample in-place
    main_params = theta[:, 0:2].clone()
    G_frame     = theta[:, 2:4].clone()
    porosity    = theta[:, 4:6].clone()
    rho_grain   = theta[:, 6:8].clone()

    # Prepare output tensor
    sim = torch.zeros(batch_size, 2, device=theta.device)

    # Track validity
    attempts = 0
    invalid_mask = torch.ones(batch_size, 2, dtype=torch.bool, device=theta.device)

    # Priors
    uniform_low, uniform_high = 0.0, 10.0
    # normals: G_frame ~ N(8.5,0.3), porosity ~ N(0.37,0.02), rho_grain ~ N(44.8,0.8)
    G_frame_mu, G_frame_std = 8.5, 0.3
    poro_mu, poro_std       = 0.37, 0.02
    grain_mu, grain_std     = 44.8, 0.8

    while invalid_mask.any() and attempts < max_attempts:
        # For entries still invalid, compute denominator and sim
        denom = main_params * (1.0 - porosity) + porosity * rho_grain  # (batch,2)
        # valid where denom>0
        valid_mask = denom > 0
        # Compute sim for valid entries
        safe_denom = denom.clamp_min(1e-12)
        sim_valid = torch.sqrt(G_frame / safe_denom)
        sim = torch.where(valid_mask, sim_valid, sim)

        # Identify still-invalid entries
        invalid_mask = ~valid_mask

        if not invalid_mask.any():
            break

        # Resample parameters for invalid entries
        idx_batch, idx_chan = invalid_mask.nonzero(as_tuple=True)
        # Uniform main params
        main_params[idx_batch, idx_chan] = torch.rand_like(main_params[idx_batch, idx_chan]) * (uniform_high - uniform_low) + uniform_low
        # G_frame
        G_frame[idx_batch, idx_chan] = torch.normal(G_frame_mu, G_frame_std, size=(len(idx_batch),), device=theta.device)
        # Porosity
        porosity[idx_batch, idx_chan] = torch.normal(poro_mu, poro_std, size=(len(idx_batch),), device=theta.device)
        # Rho_grain
        rho_grain[idx_batch, idx_chan] = torch.normal(grain_mu, grain_std, size=(len(idx_batch),), device=theta.device)

        attempts += 1

    if invalid_mask.any():
        print("Warning: maximum attempts reached, some entries remain invalid")

    return sim  # (batch_size, 2)

def simulator_full5(theta):
    """
    Simulator for batch of theta with shape (batch_size, 5):
        [ main0, main1, G_frame, porosity, rho_grain ]
    Returns:
      sim_data: tensor of shape (batch_size, 2)
                sim_data[i,0] = sqrt(G_frame[i] / denom[i,0])
                sim_data[i,1] = sqrt(G_frame[i] / denom[i,1])
      where denom[i, j] = main_j * (1 - porosity) + porosity * rho_grain.
    """
    # Ensure we have a 2D tensor
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)  # (1,5)

    batch_size, D = theta.shape
    if D != 5:
        raise ValueError(f"Expected last dim of size 5, got {D}")

    # Split out parameters
    main = theta[:, 0:2]         # (batch, 2)
    G_frame = theta[:, 2:3]      # (batch, 1)
    porosity = theta[:, 3:4]     # (batch, 1)
    rho_grain = theta[:, 4:5]    # (batch, 1)

    # Compute denominator shape (batch, 2) by broadcasting porosity & rho_grain
    denominator = main * (1.0 - porosity) + porosity * rho_grain

    # Handle invalid denominators
    # Option A: clamp them to a small positive value
    eps = 1e-12
    denom_safe = denominator.clamp_min(eps)

    # Compute simulated data
    sim_data = torch.sqrt(G_frame / denom_safe)  # broadcasts to (batch,2)

    # Optionally: mark truly invalid entries as NaN instead of clamping
    # invalid_mask = denominator <= 0
    # sim_data = torch.where(invalid_mask, torch.tensor(float('nan'), device=theta.device), sim_data)

    return sim_data  # always defined, shape (batch,2)