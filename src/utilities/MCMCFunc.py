import pints
import torch
import numpy as np
from utilities.Gassmann import *


class MCSamplingGassmannProb(pints.LogPDF):
    def __init__(self, d_obs, sigma):
        self._d_obs = torch.tensor(d_obs, dtype=torch.float32)  # Convert to torch tensor
        self._sigma = torch.tensor(sigma, dtype=torch.float32)  # Convert to torch tensor

    def __call__(self, theta):
        return self.log_pdf(theta)

    def log_pdf(self, theta):
        # Convert theta to torch tensor
        theta_torch = torch.tensor(theta, dtype=torch.float32)

        # Simulate data using the torch-based simulator
        d_pred,n = simulator_prob(theta_torch)  # Output is a torch tensor

        # Compute likelihood in torch
        likelihood = -0.5 * torch.sum(((self._d_obs - d_pred) / self._sigma) ** 2)

        # Compute prior (staying in numpy for this)
        prior = self.prior(theta, n)

        # Convert likelihood to NumPy scalar for compatibility with pints
        return likelihood.item() + prior

    def n_parameters(self):
        return len(self._d_obs)

    def prior(self, theta, m):
        # Apply prior for the main variable of interest (uniform over range [0, 10] for all elements of theta)
        main_prior = 0 if np.all((theta >= 0) & (theta <= 10)) else -np.inf

        # Priors on latent variables (stays in numpy)
        latent_prior = -0.5 * ((m[:,0] - 8.5) / 0.3) ** 2  # G_frame prior
        latent_prior += -0.5 * ((m[:,1] - 0.37) / 0.02) ** 2  # Porosity prior
        latent_prior += -0.5 * ((m[:,2] - 44.8) / 0.8) ** 2  # Rho_grain prior

        # Combine priors
        return main_prior + latent_prior
    

class MCSamplingGassmannIndepProb(pints.LogPDF):
    def __init__(self, d_obs, sigma):
        self._d_obs = torch.tensor(d_obs, dtype=torch.float32)  # Convert to torch tensor
        self._sigma = torch.tensor(sigma, dtype=torch.float32)  # Convert to torch tensor

    def __call__(self, theta):
        return self.log_pdf(theta)

    def log_pdf(self, theta):
        # Convert theta to torch tensor
        theta_torch = torch.tensor(theta, dtype=torch.float32)

        # Simulate data using the torch-based simulator
        d_pred,m = simulator_prob_indep(theta_torch)  # Output is a torch tensor

        # Compute likelihood in torch
        likelihood = -0.5 * torch.sum(((self._d_obs - d_pred) / self._sigma) ** 2)

        # Compute prior (staying in numpy for this)
        prior = self.prior(theta, m)

        # Convert likelihood to NumPy scalar for compatibility with pints
        return likelihood.item() + prior

    def n_parameters(self):
        return len(self._d_obs)

    def prior(self, theta, m):
        theta = np.asarray(theta)
        # Uniform prior on main θs:
        if not np.all((theta >= 0) & (theta <= 10)):
            return -np.inf
        main_logp = 0.0

        # Convert m to NumPy so we can index easily
        if torch.is_tensor(m):
            m = m.detach().cpu().numpy()

        # m can be (2,3) for a single eval, or (1,2,3) if batched with batch_size=1:
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]  # drop the batch dim → (2,3)
        elif m.ndim != 2 or m.shape[1] != 3:
            raise ValueError(f"Unexpected m shape: {m.shape}")

        # Now m[c, 0] = G_frame_c, m[c,1]=poro_c, m[c,2]=rho_grain_c
        mu = [8.5, 0.37, 44.8]
        sigma = [0.3, 0.02, 0.8]

        # Compute log-prior for each channel and each nuisance param, then sum
        logp_nuis = 0.0
        for c in range(2):
            for idx in range(3):
                x = m[c, idx]
                m0 = mu[idx]
                s0 = sigma[idx]
                # Gaussian log-pdf for this one x
                logp_nuis += -0.5 * ((x - m0) / s0) ** 2
                logp_nuis += -np.log(s0) - 0.5 * np.log(2 * np.pi)

        # Return a single float
        return main_logp + logp_nuis


    


class MCSamplingGassmannDet(pints.LogPDF):
    def __init__(self, d_obs, sigma):
        self._d_obs = torch.tensor(d_obs, dtype=torch.float32)  # Convert to torch tensor
        self._sigma = torch.tensor(sigma, dtype=torch.float32)  # Convert to torch tensor

    def __call__(self, theta):
        return self.log_pdf(theta)

    def log_pdf(self, theta):
        # Convert theta to torch tensor
        theta_torch = torch.tensor(theta, dtype=torch.float32)

        # Simulate data using the torch-based simulator
        d_pred = simulator_det(theta_torch)  # Output is a torch tensor

        # Compute likelihood in torch
        likelihood = -0.5 * torch.sum(((self._d_obs - d_pred) / self._sigma) ** 2)

        prior = self.prior(theta)

        # Convert likelihood to NumPy scalar for compatibility with pints
        return likelihood.item() + prior

    def n_parameters(self):
        return len(self._d_obs)

    def prior(self, theta):
        # Apply prior for the main variable of interest (uniform over range [0, 10] for all elements of theta)
        main_prior = 0 if np.all((theta >= 0) & (theta <= 10)) else -np.inf

        # Combine priors
        return main_prior


class FullMCMC(pints.LogPDF):
    def __init__(self, d_obs, sigma):
        self._d_obs = torch.tensor(d_obs, dtype=torch.float32)  # Convert to torch tensor
        self._sigma = torch.tensor(sigma, dtype=torch.float32)  # Convert to torch tensor

    def __call__(self, theta):
        return self.log_pdf(theta)

    def log_pdf(self, theta):
        # Convert theta to torch tensor
        theta_torch = torch.tensor(theta, dtype=torch.float32)

        # Simulate data using the torch-based simulator
        d_pred = simulator_full5(theta_torch)  # Output is a torch tensor

        # Compute likelihood in torch
        likelihood = -0.5 * torch.sum(((self._d_obs - d_pred) / self._sigma) ** 2)

        # Compute prior
        prior = self.prior(theta)

        # Convert likelihood to NumPy scalar for compatibility with pints
        return likelihood.item() + prior

    def n_parameters(self):
        return len(self._d_obs) + 3  # Include latent variables

    def prior(self, theta):
        # Apply prior for all elements of theta (uniform over range [0, 10])
        main_prior = 0 if np.all((theta[:len(self._d_obs)] >= 0) & (theta[:len(self._d_obs)] <= 10)) else -np.inf

        # Priors on latent variables
        latent_prior = -0.5 * ((theta[-3] - 8.5) / 0.3) ** 2  # G_frame prior
        latent_prior += -0.5 * ((theta[-2] - 0.37) / 0.02) ** 2  # Porosity prior
        latent_prior += -0.5 * ((theta[-1] - 44.8) / 0.8) ** 2  # Rho_grain prior

        # Combine priors
        return main_prior + latent_prior