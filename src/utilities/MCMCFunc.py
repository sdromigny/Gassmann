import pints
import torch
import numpy as np
from utilities.Gassmann import *

import numpy as np
import torch

import math
import time
import pints





class MCSamplingGassmannDet(pints.LogPDF):
    def __init__(self, d_obs, sigma):
        self._d_obs = torch.tensor(d_obs, dtype=torch.float32)  
        self._sigma = torch.tensor(sigma, dtype=torch.float32)  

    def __call__(self, theta):
        return self.log_pdf(theta)

    def log_pdf(self, theta):
        # Convert theta to torch tensor
        theta_torch = torch.tensor(theta, dtype=torch.float32)

        # Simulate data using the torch-based simulator
        d_pred = simulator_det(theta_torch)  

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



class PseudoMarginalMCMCController:

    def __init__(self, model, nchains, xs, method=None):
        """
        model: object providing simulator_prob, log_prior_theta, d_obs, sigma
        nchains: int
        xs: list of initial positions (length nchains) each an array-like of dimension d
        """
        self.model = model
        self.nchains = int(nchains)
        self.xs = [np.asarray(x, dtype=float).copy() for x in xs]
        if len(self.xs) != self.nchains:
            raise ValueError("xs must contain nchains initial positions")
        self._max_iterations = None
        # default proposal scale (std dev)
        self.prop_scale = 0.1
        self.report_interval = 1000

    def set_max_iterations(self, n):
        self._max_iterations = int(n)

    def set_prop_scale(self, s):
        """Set Gaussian RW scale (scalar or array-like of length dim)."""
        self.prop_scale = s

    def set_report_interval(self, n):
        self.report_interval = int(n)

    # --- internal helpers ---
    @staticmethod
    def _torch_log_likelihood_from_pred(d_obs, d_pred, sigma):
        # compute -0.5 * sum(((d_obs - d_pred)/sigma)^2)
        d_obs_t = torch.tensor(d_obs, dtype=torch.float32)
        resid = (d_obs_t - d_pred) / sigma
        return float(-0.5 * torch.sum(resid**2).item())

    # --- main run method ---
    def run(self, niter=None, rng_seed=None, verbose=True):
        """
        Run the pseudo-marginal MH sampler.

        Returns:
          chains: numpy array shape (nchains, niter, dim)
        """
        if niter is None:
            if self._max_iterations is None:
                raise ValueError("Specify niter or call set_max_iterations(...) before run()")
            niter = self._max_iterations
        else:
            niter = int(niter)

        if rng_seed is not None:
            np.random.seed(int(rng_seed))
            torch.manual_seed(int(rng_seed))

        dim = self.xs[0].shape[0]
        # handle prop_scale
        if np.isscalar(self.prop_scale):
            prop_scale_vec = np.ones(dim) * float(self.prop_scale)
        else:
            prop_scale_vec = np.asarray(self.prop_scale, dtype=float)

        chains = np.zeros((self.nchains, niter, dim), dtype=float)
        accept_counts = np.zeros(self.nchains, dtype=int)
        times = np.zeros(self.nchains, dtype=float)

        # Unpack model pieces (must exist)
        simulator_prob = getattr(self.model, "simulator_prob", None)
        log_prior_theta = getattr(self.model, "log_prior_theta", None)
        d_obs = getattr(self.model, "d_obs", None)
        sigma = getattr(self.model, "sigma", None)

        if simulator_prob is None or log_prior_theta is None or d_obs is None or sigma is None:
            raise ValueError("model must provide simulator_prob, log_prior_theta, d_obs, sigma")

        # run each chain serially
        for c in range(self.nchains):
            t0 = time.time()
            theta_curr = self.xs[c].copy()
            # Initial auxiliary sample for current theta (K = 1)
            d_pred_curr, n_curr = simulator_prob(torch.tensor(theta_curr, dtype=torch.float32))
            loglike_curr = self._torch_log_likelihood_from_pred(d_obs, d_pred_curr, sigma)
            logprior_curr = float(log_prior_theta(theta_curr))
            loghat_curr = loglike_curr            # K=1 => loghat = loglike
            logtarget_curr = logprior_curr + loghat_curr

            for t in range(niter):
                # propose
                theta_prop = theta_curr + np.random.normal(scale=prop_scale_vec, size=dim)

                # evaluate proposed unbiased estimator (one draw of nuisance)
                d_pred_prop, n_prop = simulator_prob(torch.tensor(theta_prop, dtype=torch.float32))
                loglike_prop = self._torch_log_likelihood_from_pred(d_obs, d_pred_prop, sigma)
                logprior_prop = float(log_prior_theta(theta_prop))
                loghat_prop = loglike_prop
                logtarget_prop = logprior_prop + loghat_prop

                # symmetric proposal => MH log acceptance = logtarget_prop - logtarget_curr
                logA = logtarget_prop - logtarget_curr
                if math.log(np.random.rand()) < logA:
                    # accept: replace both theta and the auxiliary/estimator
                    theta_curr = theta_prop
                    n_curr = n_prop
                    logtarget_curr = logtarget_prop
                    loghat_curr = loghat_prop
                    logprior_curr = logprior_prop
                    accept_counts[c] += 1

                chains[c, t, :] = theta_curr

                # optional progress printing
                if verbose and ((t + 1) % self.report_interval == 0):
                    acc_rate = accept_counts[c] / (t + 1)
                    print(f"chain {c+1} iter {t+1}/{niter} acc_rate={acc_rate:.3f}")

            times[c] = time.time() - t0
            if verbose:
                acc_rate = accept_counts[c] / niter
                print(f"Chain {c+1}/{self.nchains}: time {times[c]:.1f}s, accept rate {acc_rate:.3f}")

        return chains

# Example log prior on theta
def example_log_prior(theta):
    # uniform [0,10] on each element
    if np.all((theta >= 0) & (theta <= 10)):
        return 0.0
    else:
        return -np.inf

# wrap into a model-like object expected by the controller
class ExampleModel:
    def __init__(self, d_obs, sigma):
        self.d_obs = np.asarray(d_obs, dtype=float)
        self.sigma = sigma
    def simulator_prob(self, theta_torch):
        return simulator_prob(theta_torch)
    def log_prior_theta(self, theta):
        return example_log_prior(theta)
