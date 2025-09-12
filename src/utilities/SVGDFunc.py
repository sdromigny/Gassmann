import numpy as np
import torch

from scipy.spatial.distance import pdist, squareform
from utilities.Gassmann import *  
from typing import Tuple


## SVGD and sSVGD implementation based on the vip package (Zhang & Curtis (2023))
class SVGDGassmannDet:
    def __init__(self, d_obs: np.ndarray, sigma: float, device: torch.device):
        """
        d_obs   : 1D NumPy array of observed data (length = n_data)
        sigma   : observation noise std (float)
        device  : torch.device("cuda") or torch.device("cpu")
        """
        # Store d_obs and sigma as torch tensors on the correct device
        self.device = device
        self.d_obs_torch = torch.tensor(d_obs, dtype=torch.float32, device=device)
        self.sigma_torch = torch.tensor(sigma, dtype=torch.float32, device=device)

    def lnprob(self, theta_np: np.ndarray) -> np.ndarray:
        """
        Compute ∇_θ log p(θ | d_obs) for a batch of particles.

        Input:
          theta_np: NumPy array of shape (n_particles, n_params)
        Output:
          grads_np: NumPy array of shape (n_particles, n_params)
        """
        # Convert to torch with gradients enabled
        theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=self.device, requires_grad=True)
        n_particles, n_params = theta_np.shape

        # 1) Simulate predicted data, shape (n_particles, n_data)
        d_pred = simulator_det(theta_torch)  # must return a torch.Tensor on same device

        # 2) Compute log-likelihood per particle:
        #      −½ * Σ_j [ (d_obs[j] − d_pred[j]) / σ ]²
        diff = (self.d_obs_torch.unsqueeze(0) - d_pred) / self.sigma_torch
        log_likelihood = -0.5 * torch.sum(diff * diff, dim=1)  # shape: (n_particles,)

        # 3) Compute log-prior: Uniform(0,10) on each θ_i
        inside_mask = (theta_torch >= 0.0) & (theta_torch <= 10.0)  # (n_particles, n_params)
        valid_mask = torch.all(inside_mask, dim=1)  # (n_particles,)
        # log_prior = 0 where valid_mask=True, -inf where False
        log_prior = torch.where(
            valid_mask,
            torch.zeros(n_particles, device=self.device),
            torch.full((n_particles,), float("-inf"), device=self.device),
        )

        # 4) Combine to get log-posterior
        log_posterior = log_likelihood + log_prior  # (n_particles,)

        # 5) Backprop to get gradient for each particle
        grads = []
        for i in range(n_particles):
            # Zero any existing grads
            if theta_torch.grad is not None:
                theta_torch.grad.zero_()
            # Backprop just the i-th log_posterior
            log_posterior[i].backward(retain_graph=True)
            grad_i = theta_torch.grad[i].detach().cpu().numpy()  # to NumPy
            grads.append(grad_i)

        grads_np = np.stack(grads, axis=0)  # shape: (n_particles, n_params)
        return grads_np


    def svgd_kernel(self, theta: np.ndarray, h: float = -1) -> Tuple[np.ndarray, np.ndarray]:

        """
        Compute RBF kernel matrix K(x_i, x_j) and its ∇_θ K term.
        theta:   (n_particles, n_params)
        h < 0:   use median trick for bandwidth
        Returns: (Kxy [n×n], dxK [n×n_params])
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2

        if h < 0:
            m = np.median(pairwise_dists)
            h = np.sqrt(0.5 * m / np.log(theta.shape[0] + 1 + 1e-16))

        Kxy = np.exp(-pairwise_dists / (h ** 2) / 2)  # (n×n)
        dxK = -Kxy.dot(theta)  # broadcast (n×n)·(n×p) → (n×p)
        sumK = np.sum(Kxy, axis=1)  # (n,)

        # add θ_i * Σ_j K_{ij} for each dimension
        for dim_i in range(theta.shape[1]):
            dxK[:, dim_i] += theta[:, dim_i] * sumK

        dxK = dxK / (h ** 2)
        return Kxy, dxK


    def update(
        self,
        x0: np.ndarray,
        n_iter: int = 2000,
        stepsize: float = 1e-3,
        bandwidth: float = -1,
        alpha: float = 0.9,
        fudge: float = 1e-6,
        debug: bool = False,
        track_history: bool = False,
    ) -> np.ndarray:
        theta = x0.copy()  # (n_particles, n_params)
        history = []

        # If you want the initial state included:
        if track_history:
            history.append(theta.copy())

        historical_grad = np.zeros_like(theta)
        for it in range(n_iter):
            if debug and (it + 1) % 100 == 0:
                print(f"SVGD iter {it+1}/{n_iter}")

            grad_logp = self.lnprob(theta)                # (n_particles, n_params)
            Kxy, dxK = self.svgd_kernel(theta, h=bandwidth)
            phi = (Kxy.dot(grad_logp) + dxK) / theta.shape[0]

            if it == 0:
                historical_grad = phi ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (phi ** 2)

            adj_grad = phi / (fudge + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

            if track_history:
                history.append(theta.copy())

        if track_history:
            # Return a NumPy array of shape (n_iter+1, n_particles, n_params)
            return np.stack(history, axis=0)
        else:
            # Return only final particles of shape (n_particles, n_params)
            return theta
        





class SVGDGassmannProb:
    def __init__(self, d_obs: np.ndarray, sigma: float, device: torch.device):
        """
        d_obs   : 1D NumPy array of observed data (length = n_data).
        sigma   : observation noise std (float).
        device  : torch.device("cuda") or torch.device("cpu").
        """
        self.device = device
        self.d_obs_torch = torch.tensor(d_obs, dtype=torch.float32, device=device)
        self.sigma_torch = torch.tensor(sigma, dtype=torch.float32, device=device)

    def lnprob(self, theta_np: np.ndarray) -> np.ndarray:
        """
        Compute ∇_θ log p(θ | d_obs) for a batch of particles, using one
        nuisance draw m ~ p(m) per θ‐particle.

        Input:
          theta_np: NumPy array, shape = (n_particles, n_theta).
        Output:
          grads_np: NumPy array, shape = (n_particles, n_theta).
        """
        # 1) Convert to torch with requires_grad
        theta_torch = torch.tensor(
            theta_np, dtype=torch.float32, device=self.device, requires_grad=True
        )
        n_particles, n_theta = theta_np.shape

        # 2) Sample one set of nuisance parameters m per θ-particle (NumPy → torch)
        #    sample_nuis_parameters_numpy returns shape (n_particles, n_m)
        m_np = sample_nuis_parameters_numpy(n_particles)  # shape = (n_particles, n_m)
        m_torch = (
            torch.tensor(m_np, dtype=torch.float32, device=self.device).detach()
        )

        # 3) Simulate d_pred = f(θ, m)
        #    Expect shape (n_particles, n_data)
        d_pred,n = simulator_prob(theta_torch)

        # 4) Log-likelihood:  -0.5 * Σ_j [ (d_obs[j] − d_pred[j]) / σ ]²
        diff = (self.d_obs_torch.unsqueeze(0) - d_pred) / self.sigma_torch
        log_likelihood = -0.5 * torch.sum(diff * diff, dim=1)  # (n_particles,)

        # 5) Log‐prior on θ: Uniform(0,10) for each θ_i
        inside_mask = (theta_torch >= 0.0) & (theta_torch <= 10.0)
        valid_mask = torch.all(inside_mask, dim=1)  # (n_particles,)
        log_prior_theta = torch.where(
            valid_mask,
            torch.zeros(n_particles, device=self.device),
            torch.full((n_particles,), float("-inf"), device=self.device),
        )

        # 6) Log‐prior on m:  Normal(8.5,0.3) × Normal(0.37,0.02) × Normal(44.8,0.8)
        #    Each m_torch[i] = [G_frame, poro, rho_grain]
        #    => log p(m[i]) = sum of ℓ = −0.5*((m−μ)/σ)²  (ignoring constants)
        mu = torch.tensor([8.5, 0.37, 44.8], device=self.device)
        sd = torch.tensor([0.3, 0.02, 0.8], device=self.device)
        diff_m = (m_torch - mu) / sd  # shape: (n_particles, 3)
        log_prior_m = -0.5 * torch.sum(diff_m * diff_m, dim=1)  # (n_particles,)

        # 7) Combine to get log‐posterior for each particle
        log_post = log_likelihood + log_prior_theta + log_prior_m  # shape: (n_particles,)

        # 8) Compute ∇_θ[log_post] per particle
        grads = []
        for i in range(n_particles):
            if theta_torch.grad is not None:
                theta_torch.grad.zero_()
            log_post[i].backward(retain_graph=True)
            grad_i = theta_torch.grad[i].detach().cpu().numpy()  # (n_theta,)
            grads.append(grad_i)

        grads_np = np.stack(grads, axis=0)  # (n_particles, n_theta)
        return grads_np

    def svgd_kernel(self, theta: np.ndarray, h: float = -1) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute RBF kernel matrix K and its ∇_θ K term for SVGD update.

        theta: (n_particles, n_theta)
        h < 0: use median trick to choose bandwidth
        Returns:
          Kxy : (n_particles, n_particles)
          dxK : (n_particles, n_theta)
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2

        if h < 0:
            m = np.median(pairwise_dists)
            h = np.sqrt(0.5 * m / np.log(theta.shape[0] + 1 + 1e-16))

        Kxy = np.exp(-pairwise_dists / (h**2) / 2)  # (n,n)
        dxK = -Kxy.dot(theta)                       # (n,n_theta)
        sumK = np.sum(Kxy, axis=1)                  # (n,)

        # Add θ_i * Σ_j K_{ij} for each parameter dimension
        for dim_i in range(theta.shape[1]):
            dxK[:, dim_i] += theta[:, dim_i] * sumK

        dxK = dxK / (h**2)
        return Kxy, dxK

    def update(
        self,
        x0: np.ndarray,
        n_iter: int = 2000,
        stepsize: float = 1e-3,
        bandwidth: float = -1,
        alpha: float = 0.9,
        fudge: float = 1e-6,
        debug: bool = False,
        track_history: bool = False,
    ) -> np.ndarray:
        """
        Run SVGD for `n_iter` iterations, starting from initial particles x0.

        x0:         (n_particles, n_theta) initial θ samples (NumPy).
        n_iter:     number of SVGD update steps.
        stepsize:   step size for each update.
        bandwidth:  RBF kernel bandwidth h; if < 0, median trick is used.
        alpha:      momentum for AdaGrad.
        fudge:      small constant for numerical stability in AdaGrad.
        debug:      if True, prints progress every 100 iterations.
        track_history: if True, returns an array of shape
                       (n_iter+1, n_particles, n_theta), otherwise (n_particles, n_theta).

        Returns:
          If track_history=False:  (n_particles, n_theta) final θ samples.
          If track_history=True:   (n_iter+1, n_particles, n_theta) all θ at each iteration.
        """
        theta = x0.copy()  # (n_particles, n_theta)
        history = []

        if track_history:
            history.append(theta.copy())

        historical_grad = np.zeros_like(theta)
        for it in range(n_iter):
            if debug and ((it + 1) % 100 == 0):
                print(f"SVGD iter {it+1}/{n_iter}")

            # 1) ∇_θ log p(θ|d_obs) for each particle
            grad_logp = self.lnprob(theta)  # (n_particles, n_theta)

            # 2) Compute kernel matrix and ∇_θ K
            Kxy, dxK = self.svgd_kernel(theta, h=bandwidth)

            # 3) SVGD direction
            phi = (Kxy.dot(grad_logp) + dxK) / theta.shape[0]

            # 4) AdaGrad‐style scaling
            if it == 0:
                historical_grad = phi**2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (phi**2)

            adj_grad = phi / (fudge + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

            if track_history:
                history.append(theta.copy())

        if track_history:
            return np.stack(history, axis=0)
        else:
            return theta



class sSVGDGassmannProb:
    def __init__(self, d_obs: np.ndarray, sigma: float, device: torch.device):
        self.device = device
        self.d_obs_torch = torch.tensor(d_obs, dtype=torch.float32, device=device)
        self.sigma_torch = torch.tensor(sigma, dtype=torch.float32, device=device)

    def lnprob(self, theta_np: np.ndarray) -> np.ndarray:
        """
        Compute ∇_θ log p(θ | d_obs) for each θ‐particle in `theta_np`.
        Uses a single draw of nuisance m ~ p(m) per θ.
        Returns a (n_particles, n_theta) array of gradients.
        """
        theta_torch = torch.tensor(theta_np, dtype=torch.float32, device=self.device, requires_grad=True)
        n_particles, n_theta = theta_np.shape



        # 2) Simulate d_pred = simulator_prob(θ, m)
        d_pred,n = simulator_prob(theta_torch)                   # shape = (n_particles, n_data)

        # 3) log-likelihood for each particle
        diff = (self.d_obs_torch.unsqueeze(0) - d_pred) / self.sigma_torch
        log_likelihood = -0.5 * torch.sum(diff * diff, dim=1)           # (n_particles,)

        # 4) Uniform(0,10) prior on θ
        inside_mask = (theta_torch >= 0.0) & (theta_torch <= 10.0)       # (n_particles, n_theta)
        valid_mask = torch.all(inside_mask, dim=1)                      # (n_particles,)
        log_prior_theta = torch.where(
            valid_mask,
            torch.zeros(n_particles, device=self.device),
            torch.full((n_particles,), float("-inf"), device=self.device),
        )
        log_prior_n=sample_and_log_gaussians(n)

        # 6) Combine log-posteriors
        log_post = log_likelihood + log_prior_theta + log_prior_n      # (n_particles,)

        # 7) Compute ∇_θ log-posteriors via PyTorch autograd
        grads = []
        for i in range(n_particles):
            if theta_torch.grad is not None:
                theta_torch.grad.zero_()
            log_post[i].backward(retain_graph=True)
            grad_i = theta_torch.grad[i].detach().cpu().numpy()            # shape = (n_theta,)
            grads.append(grad_i)

        return np.stack(grads, axis=0)  # (n_particles, n_theta)

    def svgd_kernel(self, theta: np.ndarray, h: float = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard RBF kernel + its ∇ w.r.t. θ.
        Returns Kxy (n×n) and dxK (n×n_theta).
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2

        if h < 0:
            m = np.median(pairwise_dists)
            h = np.sqrt(0.5 * m / np.log(theta.shape[0] + 1 + 1e-16))

        Kxy = np.exp(-pairwise_dists / (h**2) / 2)
        dxK = -Kxy.dot(theta)
        sumK = np.sum(Kxy, axis=1)

        for d in range(theta.shape[1]):
            dxK[:, d] += theta[:, d] * sumK

        dxK = dxK / (h**2)
        return Kxy, dxK

    def update(
        self,
        x0: np.ndarray,
        n_iter: int = 2000,
        stepsize: float = 1e-3,
        bandwidth: float = -1,
        alpha: float = 0.9,
        fudge: float = 1e-6,
        debug: bool = False,
        track_history: bool = False,
    ) -> np.ndarray:
        """
        Run full‐batch SVGD for n_iter steps over all `N = x0.shape[0]` particles.
        Returns either the final (N, n_theta) or, if track_history=True, the full history:
          (n_iter+1, N, n_theta).
        """
        theta = x0.copy()
        history = []
        if track_history:
            history.append(theta.copy())

        historical_grad = np.zeros_like(theta)
        for it in range(n_iter):
            if debug and (it + 1) % 100 == 0:
                print(f"SVGD iter {it+1}/{n_iter}")

            # 1) ∇_θ log p(θ|d) over **all** N particles
            grad_logp = self.lnprob(theta)                 # (N, n_theta)

            # 2) Full‐batch kernel on all N particles
            Kxy, dxK = self.svgd_kernel(theta, h=bandwidth)

            # 3) φ = (K⋅∇logp + dxK) / N
            phi = (Kxy.dot(grad_logp) + dxK) / theta.shape[0]

            # 4) AdaGrad‐style
            if it == 0:
                historical_grad = phi**2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (phi**2)

            adj_grad = phi / (fudge + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

            if track_history:
                history.append(theta.copy())

        if track_history:
            return np.stack(history, axis=0)
        else:
            return theta




class sSVGDGassmannDet:
    def __init__(self, d_obs: np.ndarray, sigma: float, device: torch.device):
        self.device = device
        self.d_obs_torch = torch.tensor(d_obs, dtype=torch.float32, device=device)
        self.sigma_torch = torch.tensor(sigma, dtype=torch.float32, device=device)

    def lnprob(self, theta_np: np.ndarray) -> np.ndarray:
        theta = torch.tensor(theta_np,
                            dtype=torch.float32,
                            device=self.device,
                            requires_grad=True)

        # 1) forward simulate
        d_pred = simulator_det_cuda(theta)   # now fully vectorized

        # 2) log-likelihood
        diff = (self.d_obs_torch.unsqueeze(0) - d_pred) / self.sigma_torch
        log_lik = -0.5 * diff.pow(2).sum(dim=1)

        # 3) uniform prior
        inside = (theta >= 0) & (theta <= 10)
        log_prior = torch.where(inside.all(dim=1),
                                0.0,
                                float("-inf"))

        log_post = log_lik + log_prior

        # 4) get all grads in *one* pass
        grads_torch, = torch.autograd.grad(
            log_post.sum(),
            theta,
            retain_graph=False,
            create_graph=False
        )

        return grads_torch.detach().cpu().numpy()


    def svgd_kernel(self, theta: np.ndarray, h: float = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard RBF kernel + its ∇ w.r.t. θ.
        Returns Kxy (n×n) and dxK (n×n_theta).
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2

        if h < 0:
            m = np.median(pairwise_dists)
            h = np.sqrt(0.5 * m / np.log(theta.shape[0] + 1 + 1e-16))

        Kxy = np.exp(-pairwise_dists / (h**2) / 2)
        dxK = -Kxy.dot(theta)
        sumK = np.sum(Kxy, axis=1)

        for d in range(theta.shape[1]):
            dxK[:, d] += theta[:, d] * sumK

        dxK = dxK / (h**2)
        return Kxy, dxK

    def update(
        self,
        x0: np.ndarray,
        n_iter: int = 2000,
        stepsize: float = 1e-3,
        bandwidth: float = -1,
        alpha: float = 0.9,
        fudge: float = 1e-6,
        debug: bool = False,
        track_history: bool = False,
    ) -> np.ndarray:
        """
        Run full‐batch SVGD for n_iter steps over all `N = x0.shape[0]` particles.
        Returns either the final (N, n_theta) or, if track_history=True, the full history:
          (n_iter+1, N, n_theta).
        """
        theta = x0.copy()
        history = []
        if track_history:
            history.append(theta.copy())

        historical_grad = np.zeros_like(theta)
        for it in range(n_iter):
            if debug and (it + 1) % 100 == 0:
                print(f"SVGD iter {it+1}/{n_iter}")

            # 1) ∇_θ log p(θ|d) over **all** N particles
            grad_logp = self.lnprob(theta)                 # (N, n_theta)

            # 2) Full‐batch kernel on all N particles
            Kxy, dxK = self.svgd_kernel(theta, h=bandwidth)

            # 3) φ = (K⋅∇logp + dxK) / N
            phi = (Kxy.dot(grad_logp) + dxK) / theta.shape[0]

            # 4) AdaGrad‐style
            if it == 0:
                historical_grad = phi**2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (phi**2)

            adj_grad = phi / (fudge + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

            if track_history:
                history.append(theta.copy())

        if track_history:
            return np.stack(history, axis=0)
        else:
            return theta






class FullsSVGD:
    def __init__(self, d_obs: np.ndarray, sigma: float, device: torch.device):
        self.device        = device
        # observed data
        self.d_obs_torch   = torch.tensor(d_obs, dtype=torch.float32, device=device)
        # known noise std
        self.sigma_torch   = torch.tensor(sigma, dtype=torch.float32, device=device)

        # latent prior hyper-parameters
        self.latent_mu     = torch.tensor([8.5, 0.37, 44.8], dtype=torch.float32, device=device)
        self.latent_sd     = torch.tensor([0.3, 0.02, 0.8], dtype=torch.float32, device=device)

    def lnprob(self, theta_np: np.ndarray) -> np.ndarray:
        """
        Returns ∇_θ log p(θ | d_obs) for each θ‐particle in `theta_np`.
        """
        theta = torch.tensor(theta_np,
                            dtype=torch.float32,
                            device=self.device,
                            requires_grad=True)  # (N,D)

        # 2) simulate forward
        d_pred = simulator_full5(theta)  # -> (N, len(d_obs))

        # 3) log‐likelihood
        diff = (self.d_obs_torch.unsqueeze(0) - d_pred) / self.sigma_torch
        log_like = -0.5 * torch.sum(diff**2, dim=1)  # (N,)

        # 4) uniform prior on first two dims
        in_bounds = ((theta[:, :2] >= 0.0) & (theta[:, :2] <= 10.0)).all(dim=1)
        log_prior_unif = torch.where(in_bounds,
                                    torch.zeros_like(log_like),
                                    torch.full_like(log_like, float("-inf")))

        # 5) Gaussian prior on last three
        latent = theta[:, -3:]
        diff_latent = (latent - self.latent_mu) / self.latent_sd
        log_prior_latent = -0.5 * torch.sum(diff_latent**2, dim=1)  # (N,)

        # 6) total log‐posterior
        log_post = log_like + log_prior_unif + log_prior_latent  # (N,)

        # 7) **one** autograd call for the **full** Jacobian:
        #    grad_outputs[i] picks out d log_post[i]/d theta so we get an (N,D) tensor back.
        grads = torch.autograd.grad(
            outputs    = log_post,
            inputs     = theta,
            grad_outputs=torch.ones_like(log_post),
            create_graph=False,
            retain_graph=False
        )[0]  # a (N,D) tensor

        return grads.cpu().numpy()
    
    def svgd_kernel(self, theta: np.ndarray, h: float = -1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standard RBF kernel + its ∇ w.r.t. θ.
        Returns Kxy (n×n) and dxK (n×n_theta).
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2

        if h < 0:
            m = np.median(pairwise_dists)
            h = np.sqrt(0.5 * m / np.log(theta.shape[0] + 1 + 1e-16))

        Kxy = np.exp(-pairwise_dists / (h**2) / 2)
        dxK = -Kxy.dot(theta)
        sumK = np.sum(Kxy, axis=1)

        for d in range(theta.shape[1]):
            dxK[:, d] += theta[:, d] * sumK

        dxK = dxK / (h**2)
        return Kxy, dxK

    def update(
        self,
        x0: np.ndarray,
        n_iter: int = 2000,
        stepsize: float = 1e-3,
        bandwidth: float = -1,
        alpha: float = 0.9,
        fudge: float = 1e-6,
        debug: bool = False,
        track_history: bool = False,
    ) -> np.ndarray:
        """
        Run full‐batch SVGD for n_iter steps over all `N = x0.shape[0]` particles.
        Returns either the final (N, n_theta) or, if track_history=True, the full history:
          (n_iter+1, N, n_theta).
        """
        theta = x0.copy()
        history = []
        if track_history:
            history.append(theta.copy())

        historical_grad = np.zeros_like(theta)
        for it in range(n_iter):
            if debug and (it + 1) % 100 == 0:
                print(f"SVGD iter {it+1}/{n_iter}")

            # 1) ∇_θ log p(θ|d) over **all** N particles
            grad_logp = self.lnprob(theta)                 # (N, n_theta)

            # 2) Full‐batch kernel on all N particles
            Kxy, dxK = self.svgd_kernel(theta, h=bandwidth)

            # 3) φ = (K⋅∇logp + dxK) / N
            phi = (Kxy.dot(grad_logp) + dxK) / theta.shape[0]

            # 4) AdaGrad‐style
            if it == 0:
                historical_grad = phi**2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (phi**2)

            adj_grad = phi / (fudge + np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad

            if track_history:
                history.append(theta.copy())

        if track_history:
            return np.stack(history, axis=0)
        else:
            return theta



