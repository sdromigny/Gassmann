import numpy as np
import torch
import torch.distributions.transforms as T

import h5py
import os
from utilities.NormFlows import *


import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from utilities.Gassmann import * 
from typing import Tuple
def lnprob_factory(forward_model, d_obs, sigma, device, prior=None):
    """
    Returns a function lnprob(z_np) compatible with sSVGD.

    z_np: unconstrained latent particles, shape (n_particles, 5)
    returns:
        logp: (n_particles,)
        grad: (n_particles, 5)
        None
    """

    d_obs_t = torch.as_tensor(d_obs, dtype=torch.float64, device=device)

    def lnprob(z_np):
        # Accept numpy or torch input
        if isinstance(z_np, np.ndarray):
            z = torch.as_tensor(z_np, dtype=torch.float64, device=device)
        else:
            z = z_np.to(device=device, dtype=torch.float64)

        z = z.clone().detach().requires_grad_(True)

        # Transform latent -> physical parameters
        # Use the transform that matches your 5D setup
        theta, log_det = batch_constrained_transform(z)

        # Forward model
        pred = forward_model(theta)

        # Make sure shapes are (batch, 2)
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)

        resid = pred - d_obs_t
        log_likelihood = -0.5 * torch.sum((resid / sigma) ** 2, dim=1)

        # Optional prior in physical space
        if prior is None:
            log_prior = torch.zeros_like(log_likelihood)
        else:
            log_prior = prior(theta)

        logp = log_likelihood + log_det + log_prior

        grad = torch.autograd.grad(logp.sum(), z, retain_graph=False, create_graph=False)[0]

        return (
            logp.detach().cpu().numpy(),
            grad.detach().cpu().numpy(),
            None
        )

    return lnprob


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




import numpy as np
from scipy.spatial.distance import pdist, squareform

import numpy as np
from scipy.spatial.distance import pdist, squareform

def svgd_grad(x, grad, kernel='rbf', w=None, h=1.0, chunks=None):
    """
    Replacement svgd_grad: computes kernel matrix kxy and SVGD gradient sgrad directly.

    Inputs
      x      - 2D array of particles (n, d)
      grad   - 2D array of gradients of logp (n, d)
      kernel - 'rbf' supported (diagonal not implemented here)
      w      - 1D vector of weights for dimensions (length d)
      h      - bandwidth multiplier (fraction of median-trick bandwidth)
      chunks - tuple-like (nparticles, chunk_dim); if None defaults to x.shape

    Outputs
      kxy    - kernel matrix (n, n)
      sgrad  - SVGD gradient for each particle (n, d)
    """

    n, d = x.shape

    if w is None:
        w = np.ones((d,), dtype=x.dtype)
    if chunks is None:
        chunks = x.shape  # (n, d)

    # --- 1) compute weighted pairwise squared distances (condensed) in chunks ---
    dist_condensed = None  # will be length n*(n-1)/2

    chunk_width = chunks[1]
    for i in range(0, d, chunk_width):
        end = min(i + chunk_width, d)
        xs = np.ascontiguousarray(x[:, i:end])        # (n, chunk_w)
        ws = np.ascontiguousarray(w[i:end])           # (chunk_w,)
        # scale dims by sqrt(weight) to implement weighted squared distance
        xs_w = xs * np.sqrt(ws[np.newaxis, :])
        chunk_dist = pdist(xs_w, metric="sqeuclidean")  # condensed squared distances
        if dist_condensed is None:
            dist_condensed = chunk_dist
        else:
            # accumulate (pdist condensed vectors must have same length)
            dist_condensed = dist_condensed + chunk_dist

    if dist_condensed is None:
        # fallback (shouldn't happen unless d == 0)
        dist_condensed = np.zeros((n * (n - 1)) // 2, dtype=x.dtype)

    # --- 2) median trick bandwidth ---
    medh = np.median(dist_condensed)
    # guard against zero median
    if medh <= 0:
        medh = 1.0
    medh = np.sqrt(0.5 * medh / np.log(n + 1.0))
    h_actual = h * medh if h > 0 else medh

    # --- 3) kernel matrix K ---
    # note: dist_condensed are squared distances
    K_cond = np.exp(-dist_condensed / (2.0 * (h_actual ** 2)))
    K = squareform(K_cond)     # (n,n)
    np.fill_diagonal(K, 1.0)

    # --- 4) compute SVGD gradient ---
    # Term A: K @ grad  (n,d)
    A = K.dot(grad)            # (n, d)

    # Term B: gradient of kernel sum:
    # For RBF: grad_{x_j} k(x_j, x_i) = -1/h^2 * k_ji * (x_j - x_i)
    # Summing over j gives: B_i = sum_j grad_{x_j} k(x_j, x_i)
    # We compute: S = K @ X  (n,d), and row_sum = sum_j K_ij
    row_sum = np.sum(K, axis=1)              # (n,)
    S = K.dot(x)                             # (n,d)
    # B = -1/h^2 * ( S - x * row_sum[:,None] )
    scalar = -1.0 / (h_actual ** 2)
    B = scalar * (S - x * row_sum[:, None])  # (n,d)

    # Combine and normalise by n (as in SVGD estimator)
    sgrad = (A + B) / float(n)

    return K, sgrad


def batch_constrained_transform2d(w):
    """
    w: (batch_size, 2)   latent from N(0,1)
    returns:
      theta: (batch_size, 2)  in mixed constrained space
      log_det: (batch_size,)  log |det d w→θ|
    """
    batch_size = w.shape[0]
    device = w.device

    # 1) Uniform[0,10] for dims 0,1
    sigmoid = T.SigmoidTransform()  # maps ℝ → (0,1)
    u = sigmoid(w)           # shape (batch,2)
    theta_u = u * 10.0              # shape (batch,2)
    # log-det from sigmoid:
    ld_sig = sigmoid.log_abs_det_jacobian(w, u).sum(dim=1)
    # extra log-det from multiplying by 10:
    ld_scale_u = 2 * torch.log(torch.tensor(10.0, device=device))


    # assemble θ and log_det
    theta = torch.cat([theta_u], dim=1)
    log_det = ld_sig + ld_scale_u 

    return theta, log_det


class sSVGD():
    '''
    A class that implements stochastic SVGD algorithm.
    '''
    def __init__(self, lnprob, kernel='rbf', h=1.0, mask=None, threshold=0.02,
                 weight='grad', out='samples.hdf5'):
        '''
        lnprob: log of the probability density function, usually negtive misfit function
        kernel: kernel function, including rbf and diagonal matrix kernel
        h:  bandwith for rbf kernel function, is a frac of median trick bandwith.
            h is positive, actual bandwith is h*medh
        weight: method of coonstructing diagonal matrix, 'var' using variance of each parameters across particles,
            'grad' using 1/sqrt(grad**2) similar as in adagrad, 'delta' using sqrt(dg**2)/sqrt(dm**2)
        mask: mask array where the variables are fixed, i.e. the gradient is zero, default no mask
        out: hdf5 file that stores final particles

        '''

        self.h = h
        self.lnprob = lnprob
        self.kernel = kernel
        self.mask = mask
        self.out = out
        self.threshold = threshold
        self.weight = weight
        if(kernel=='rbf'):
            self.weight = 'constant'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def sample(self, x0, n_iter=1000, metropolis=False, stepsize=1e-2, gamma=1.0, decay_step=1,
               burn_in=100, thin=2, alpha=0.9, beta=0.95, chunks=None, optimizer=None):
        '''
        Using ssvgd to sample a probability density function


        # Check input
        if x0 is None :
            raise ValueError('x0 cannot be None!')

        if(chunks is None):
            chunks = x0.shape

        # create a hdf5 file to store samples on disk
        nsamples = int((n_iter-burn_in)/thin)
        if(not os.path.isfile(self.out)):
            f = h5py.File(self.out,'a')
            samples = f.create_dataset('samples',(nsamples,x0.shape[0],x0.shape[1]),
                                       maxshape=(None,x0.shape[0],x0.shape[1]),
                                       compression="gzip", chunks=True)
        else:
            f = h5py.File(self.out,'a')
            f['samples'].resize((f['samples'].shape[0]+nsamples),axis=0)
            samples = f['samples']

        # sampling
        if(metropolis):
            losses, theta = self.ma_sample(x0, samples, n_iter=n_iter, stepsize=stepsize, gamma=gamma, decay_step=decay_step,
               burn_in=burn_in, thin=thin, alpha=alpha, beta=beta, chunks=chunks, optimizer=optimizer)
        else:
            losses, theta = self.pl_sample(x0, samples, n_iter=n_iter, stepsize=stepsize, gamma=gamma, decay_step=decay_step,
               burn_in=burn_in, thin=thin, alpha=alpha, beta=beta, chunks=chunks, optimizer=optimizer)

        # close hdf5 file
        f.close()

        return losses, theta

    def grad(self, theta, mkernel=None, chunks=None):
        '''
        Compute gradients for ssvgd update
        Input
            theta: the current value of variable (transformed), shape (n,dim)
            mkernel: the vector of the diagonal matrix with length dim, if using a diagonal matrix kernel
            chunks: chunks of theta for calculation, default theta.shape
        Return
            logp: log posterior pdf value across particles
            sgrad: svgd gradients for each particles, shape (n,dim)
            kxy: kernel matrix, shape (n,n)
        '''

        if(mkernel is None):
            mkernel = np.full((theta.shape[1],),fill_value=1.0)

        logp, grad, _ = self.lnprob(theta)
        kxy, sgrad = svgd_grad(theta, grad, kernel=self.kernel, w=mkernel, h=self.h, chunks=chunks)
        print(f'max, mean, median, and min grads for svgd: {np.max(abs(sgrad))} {np.mean(abs(sgrad))} {np.median(abs(sgrad))} {np.min(abs(sgrad))}')

        return logp, sgrad, kxy

    def update(self, theta, step=1e-3, mkernel=None, chunks=None):
        '''
        Compute gradients for ssvgd update
        Input
            theta: the current value of variable (transformed), shape (n,dim)
            mkernel: the vector of the diagonal matrix with length dim, if using a diagonal matrix kernel
            chunks: chunks of theta for calculation, default theta.shape
        Return
            update_step: update at each iteration
            loss: mean loss value across particles
            grad: svgd gradients for each particles, shape (n,dim)
        '''

        if(mkernel is None):
            mkernel = np.full((theta.shape[1],),fill_value=1.0)

        # get svgd gradient and kernel matrix K
        logp, sgrad, kxy = self.grad(theta, mkernel=mkernel, chunks=chunks)

        # calculate cholesky decomposition of kernel matrix K and generate random variable
        cholK = np.linalg.cholesky(2*kxy/theta.shape[0])
        random_update = np.sqrt(1./mkernel)*np.matmul(cholK,np.random.normal(size=theta.shape))

        update_step = step*sgrad + np.sqrt(step)*random_update
        if(self.mask is not None):
            update_step[:,self.mask] = 0

        return update_step, logp, sgrad

    def pl_sample(self, x0, samples, n_iter=1000, stepsize=1e-2, gamma=1.0, decay_step=1,
               burn_in=100, thin=2, alpha=0.9, beta=0.95, chunks=None, optimizer=None):
        '''
        Using ssvgd to sample a probability density function
        Input
            x0: initial value, shape (n,dim)
            n_iter: number of iterations
            stepsize: stepsize for each iteration
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the stepsize
            burn_in: burn_in period
            thin: thining of the chain
            alpha, beta: hyperparameter for sgd and adam, for sgd only alpha is ued
            chunks: chunks of theta for calculation, default theta.shape
        Return
            losses: loss value for each iterations, shape (n_iter,n)
            The final particles are stored at the hdf5 file specified by self.out, so no return samples
        '''

        # Check input
        if x0 is None :
            raise ValueError('x0 cannot be None!')

        if(chunks is None):
            chunks = x0.shape

        theta = np.copy(x0).astype(np.float64)
        losses = np.zeros((n_iter,x0.shape[0]))

        # initialise some variables
        nsamples = int((n_iter-burn_in)/thin)
        sample_count = 0
        prev_grad = np.zeros(x0.shape,dtype=np.float64)
        prev_theta = np.zeros(x0.shape,dtype=np.float64)
        mkernel = np.full((theta.shape[1],),fill_value=1.0, dtype=np.float64)
        w = weight(dim=theta.shape[1], approx=self.weight, threshold=self.threshold)

        # sampling
        for i in range(n_iter):

            # start a new iteration
            print(f'Iteration: {i}')
            #print(f'max, mean, median and min kernel: {np.max(abs(mkernel))} {np.mean(abs(mkernel))} {np.median(abs(mkernel))} {np.min(abs(mkernel))}')
            print(f'max, mean, median and min theta: {np.max(abs(theta))} {np.mean(abs(theta))} {np.median(abs(theta))} {np.min(abs(theta))}')
            update_step, logp, pgrad = self.update(theta, step=stepsize, mkernel=mkernel, chunks=chunks)

            mkernel = w.diag(theta, prev_theta, pgrad, prev_grad)
            prev_grad = np.copy(pgrad)
            prev_theta = np.copy(theta)

            theta = theta + update_step
            losses[i,:] = -logp
            #print('Average loss: '+str(np.mean(-logp)))

            # decay the stepsize if required
            if((i+1)%decay_step == 0):
                stepsize = stepsize * gamma

            # after burn_in then collect samples
            if(i>=burn_in and (i-burn_in)%thin==0):
                samples[-nsamples+sample_count,:,:] = np.copy(theta)
                sample_count += 1

        return losses, theta

    def ma_sample(self, x0, samples, n_iter=1000, stepsize=1e-2, gamma=1.0, decay_step=1,
               burn_in=100, thin=2, alpha=0.9, beta=0.95, chunks=None, optimizer=None):
        '''
        Using ssvgd to sample a probability density function
        Input
            x0: initial value, shape (n,dim)
            n_iter: number of iterations
            stepsize: stepsize for each iteration
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the stepsize
            burn_in: burn_in period
            thin: thining of the chain
            alpha, beta: hyperparameter for sgd and adam, for sgd only alpha is ued
            chunks: chunks of theta for calculation, default theta.shape
        Return
            losses: mean loss value for each iterations, vector of length n
            The final particles are stored at the hdf5 file specified by self.out, so no return samples
        '''

        # Check input
        if x0 is None :
            raise ValueError('x0 cannot be None!')

        if(chunks is None):
            chunks = x0.shape

        theta = np.copy(x0).astype(np.float64)
        losses = np.zeros((n_iter,x0.shape[0]))

        # initialise some variables
        nsamples = int((n_iter-burn_in)/thin)
        sample_count = 0; accepted_count = 0
        mkernel = np.full((theta.shape[1],),fill_value=1.0, dtype=np.float64)
        logp, sgrad, kxy = self.grad(theta, mkernel=mkernel, chunks=chunks)
        cholK = np.linalg.cholesky(kxy/theta.shape[0])

        # save computed info for the current model
        prev_logp = logp
        prev_grad = sgrad
        prev_kxy = kxy
        prev_cholK = cholK
        prev_theta = theta

        # sampling
        for i in range(n_iter):

            # start a new iteration
            print(f'Iteration: {i}')
            print(f'max, mean, median and min theta: {np.max(abs(theta))} {np.mean(abs(theta))} {np.median(abs(theta))} {np.min(abs(theta))}')

            # update on the current model
            random_update = np.sqrt(1./mkernel)*np.matmul(cholK,np.random.normal(size=theta.shape))
            update_step = stepsize*sgrad + np.sqrt(2*stepsize)*random_update
            if(self.mask is not None):
                update_step[:,self.mask] = 0

            # update theta
            theta = theta + update_step

            # compute forward proposal pdf q(theta_k+1|theta_k)
            update = theta - prev_theta - stepsize*sgrad # masked variables have no effect on proposal pdf
            pvar = sla.solve_triangular(cholK,update)
            forward_plogp = -0.25/stepsize*np.sum(pvar.flatten()**2) - 0.5*sla.det(2*stepsize*kxy/theta.shape[0]) - 0.5*pvar.flatten().size*np.log(2*np.pi)

            # compute info for updated theta
            logp, sgrad, kxy = self.grad(theta, mkernel=mkernel, chunks=chunks)

            # compute reverse proposal pdf q(theta_k|theta_k+1)
            cholK = np.linalg.cholesky(kxy/theta.shape[0])
            reverse_update = prev_theta - theta - stepsize*sgrad
            pvar = sla.solve_triangular(cholK,reverse_update)
            reverse_plogp = -0.25/stepsize*np.sum(pvar.flatten()**2) - 0.5*sla.det(2*stepsize*kxy/theta.shape[0]) - 0.5*pvar.flatten().size*np.log(2*np.pi)

            # compute acceptance ratio
            acceptance_ratio = np.sum(logp) + reverse_plogp - ( np.sum(prev_logp) + forward_plogp )
            print(f'log pdf change: {np.sum(logp-prev_logp)} {reverse_plogp-forward_plogp}')
            acceptance_ratio = min(0,acceptance_ratio)

            if( np.log(np.random.uniform()) < acceptance_ratio ):
                # update previously save info if accepted
                prev_theta = theta
                prev_logp = logp
                prev_grad = sgrad
                prev_kxy = kxy
                prev_cholK = cholK
                accepted_count = accepted_count + 1
            else:
                # recover info if rejected
                theta = prev_theta
                kxy = prev_kxy
                cholK = prev_cholK
                logp = prev_logp
                sgrad = prev_grad


            losses[i,:] = -logp
            print(f'Real time and total acceptance rate: {np.exp(acceptance_ratio)} {accepted_count*1.0/(i+1)}')

            # decay the stepsize if required
            if((i+1)%decay_step == 0):
                stepsize = stepsize * gamma

            # after burn_in then collect samples
            if(i>=burn_in and (i-burn_in)%thin==0):
                samples[-nsamples+sample_count,:,:] = np.copy(theta)
                sample_count += 1

        return losses, theta

class weight():
    '''
    A class that generates a weigting vector for kernel functions
    '''

    def __init__(self, dim=100, approx='grad', alpha=0.95, beta=0.9,
                 quantile=0.8, threshold=0.02):

        self.dm = np.zeros((dim,),dtype=np.float64)
        self.dg = np.zeros((dim,),dtype=np.float64)
        self.approx = approx
        self.kernel = np.full((dim,),fill_value=1.0,dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.quantile = quantile
        self.threshold = threshold

    def diag(self, cm, pm, cg, pg, eps=1E-5):

        if(self.approx=='constant'):
            pass

        elif(self.approx=='var'):
            kernel = 1/(np.var(cm,axis=0)+eps)
            kernel[kernel<self.threshold] = self.threshold
            kernel = kernel/np.quantile(kernel,self.quantile)
            self.kernel = kernel

        elif(self.approx=='bfgs'):
            invH = 1./self.kernel
            dg = cg - pg
            dx = cm - pm
            rho = 1./np.sum(dx*dg,axis=1)
            gh = np.sum(dg**2*invH, axis=1)
            kernel = (rho**2*gh+rho)*dx**2 + invH - 2*rho*dx*dg*invH
            kernel = 1./kernel
            self.kernel = np.sqrt(np.mean(kernel**2,axis=0))

        elif(self.approx=='delta'):
            self.dm = (1-self.alpha)*self.dm + self.alpha*np.mean((cm - pm)**2,axis=0)
            self.dg = (1-self.alpha)*self.dg + self.alpha*np.mean((cg - pg)**2,axis=0)
            kernel = self.dg/(self.dm+eps)
            kernel = np.sqrt(kernel)
            kernel = kernel/np.quantile(kernel,self.quantile)
            kernel[kernel<self.threshold] = self.threshold
            self.kernel = kernel

        elif(self.approx=='grad'):
            self.dg = (1-self.alpha)*np.mean(cg**2,axis=0) + self.alpha*self.dg
            kernel = np.sqrt(self.dg)
            #kernel = kernel/np.quantile(kernel,self.quantile)
            kernel[kernel<self.threshold] = self.threshold
            self.kernel = kernel

        elif(self.approx=='adam'):
            self.dm = (1-self.alpha)*cg + self.alpha*self.dm
            self.dg = (1-self.beta)*cg**2 + self.beta*self.dg
            kernel = np.mean(self.dg,axis=0)/np.mean(np.abs(self.dm)+eps,axis=0)
            kernel = kernel/np.quantile(kernel,self.quantile)
            kernel[kernel<self.threshold] = self.threshold
            self.kernel = kernel

        return self.kernel
