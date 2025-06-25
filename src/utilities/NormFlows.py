import torch
import normflows as nf
import torch.nn as nn

def unconstrained_to_constrained(w: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """
    Map unconstrained w (real) to θ in (a, b) via scaled sigmoid:
      θ = a + (b-a) * sigmoid(w).
    w: tensor of any shape.
    Returns tensor of same shape, values in (a, b).
    """
    sig = torch.sigmoid(w)
    return a + (b - a) * sig

def transform_to_theta(w):
    # w: (batch, dim)
    return torch.sigmoid(w)  # in (0,1)


def log_det_transform(z, low=0.0, high=1.0):
    sig = torch.sigmoid(z)
    w = low + (high - low) * sig

    # Compute log|dw/dz| = log(scale * sigmoid(z) * (1 - sigmoid(z)))
    scale = high - low
    log_det_jac = (
        torch.log(torch.tensor(scale)) +
        torch.log(sig + 1e-12) +
        torch.log(1 - sig + 1e-12)
    )

    return w, torch.sum(log_det_jac, dim=1)

class NormFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.K = config['K']
        self.latent_size = config['latent_size']
        self.hidden_layers = config['hidden_layers']
        self.hidden_units = config['hidden_units']
        self.dim = config['dim']
        self.flow_type = config['flow_type']  # Ensure this is passed correctly

        # Get flow layers based on the flow_type
        self.flows = self.get_flows()

        # Initialize the base distribution (q0) and Normalizing Flow model
        self.q0 = nf.distributions.DiagGaussian(self.dim, trainable=False)
        self.nfm = nf.NormalizingFlow(q0=self.q0, flows=self.flows)


    def get_flows(self):
        """Select and return the appropriate flow layers based on flow_type."""
        if self.flow_type == 'rsqf':
            return self.rsqf()
        elif self.flow_type == 'planar':
            return self.planar()
        elif self.flow_type == 'masked':
            return self.masked()
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")

    def rsqf(self):
        """Define and return the flow layers for the Normalizing Flow model."""
        flows = []
        for i in range(self.K):
            flows.append(nf.flows.AutoregressiveRationalQuadraticSpline(self.latent_size, 
                                                                       self.hidden_layers, 
                                                                       self.hidden_units))
            flows.append(nf.flows.LULinearPermute(self.latent_size))

        return flows

    def planar(self):
        """Define and return the planar flow layers for the Normalizing Flow model."""
        flows = []
        for i in range(self.K):
            flows.append(nf.flows.Planar((self.dim,)))
        return flows
    
    def masked(self):
        """Define and return the flow layers for the Normalizing Flow model."""
        flows = []
        for i in range(self.K):
            flows.append(nf.flows.MaskedAffineAutoregressive(self.latent_size, 
                                                            self.hidden_units,  
                                                            num_blocks=2))
            flows.append(nf.flows.LULinearPermute(self.latent_size))

        return flows
    
    def forward_and_log_det(self, z):
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:  # Use flows from the normalizing flow model
            z, log_d = flow(z)
            log_det += log_d
        return z, log_det
    
    def sample(self, num_samples=1):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q




    
import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    """
    PyTorch implementation for a linear transform: z = u + Lx
    If this flow is used for ADVI, then the covariance matrix is Σ = L^T L
    """

    def __init__(self, dim, kernel='diagonal', trainable=True):
        super().__init__()
        self.dim = dim
        self.kernel = kernel

        self.u = nn.Parameter(torch.zeros(dim), requires_grad=trainable)
        self.diag = nn.Parameter(torch.zeros(dim), requires_grad=trainable)

        # Only allocate non-diagonal elements if using full-rank
        if kernel == 'fullrank':
            self.non_diag = nn.Parameter(torch.zeros(int(dim * (dim - 1) / 2)), requires_grad=trainable)
        else:
            self.non_diag = None

        # self.base_dist = torch.distributions.MultivariateNormal(
        #     loc=torch.zeros(dim), 
        #     covariance_matrix=torch.eye(dim)
        # )

    def create_lower_triangular(self, diagonal=0):
        lower = torch.zeros((self.dim, self.dim), device=self.diag.device)
        if self.kernel == 'fullrank' and self.non_diag is not None:
            indices = np.tril_indices(self.dim, diagonal)
            lower[indices] = self.non_diag
        return lower

    def forward_and_log_det(self, x, train=True):
        x = x.to(self.u.device)
        diag = torch.exp(self.diag)
        L = torch.diag(diag) + self.create_lower_triangular(diagonal=-1)
        z = self.u + torch.matmul(x, L.T)
        log_det = torch.log(diag).sum().repeat(x.shape[0])
        return z, log_det

    def sample(self, num_samples=1):
        device = self.u.device

        # Re-create base distribution on the correct device
        base_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.dim, device=device),
            covariance_matrix=torch.eye(self.dim, device=device)
        )

        x = base_dist.sample((num_samples,))
        z, log_det = self.forward_and_log_det(x)
        log_prob = base_dist.log_prob(x) - log_det  # Now everything is on same device
        return z, log_prob



    
import torch
import torch.nn as nn
import numpy as np

class Linear(nn.Module):
    """
    PyTorch implementation for a linear transform: z = u + Lx
    If this flow is used for ADVI, then the covariance matrix is Σ = L^T L
    """

    def __init__(self, dim, kernel='diagonal', trainable=True):
        super().__init__()
        self.dim = dim
        self.kernel = kernel

        self.u = nn.Parameter(torch.zeros(dim), requires_grad=trainable)
        self.diag = nn.Parameter(torch.zeros(dim), requires_grad=trainable)

        # Only allocate non-diagonal elements if using full-rank
        if kernel == 'fullrank':
            self.non_diag = nn.Parameter(torch.zeros(int(dim * (dim - 1) / 2)), requires_grad=trainable)
        else:
            self.non_diag = None

        # self.base_dist = torch.distributions.MultivariateNormal(
        #     loc=torch.zeros(dim), 
        #     covariance_matrix=torch.eye(dim)
        # )

    def create_lower_triangular(self, diagonal=0):
        lower = torch.zeros((self.dim, self.dim), device=self.diag.device)
        if self.kernel == 'fullrank' and self.non_diag is not None:
            indices = np.tril_indices(self.dim, diagonal)
            lower[indices] = self.non_diag
        return lower

    def forward_and_log_det(self, x, train=True):
        x = x.to(self.u.device)
        diag = torch.exp(self.diag)
        L = torch.diag(diag) + self.create_lower_triangular(diagonal=-1)
        z = self.u + torch.matmul(x, L.T)
        log_det = torch.log(diag).sum().repeat(x.shape[0])
        return z, log_det

    def sample(self, num_samples=1):
        device = self.u.device

        # Re-create base distribution on the correct device
        base_dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.dim, device=device),
            covariance_matrix=torch.eye(self.dim, device=device)
        )

        x = base_dist.sample((num_samples,))
        z, log_det = self.forward_and_log_det(x)
        log_prob = base_dist.log_prob(x) - log_det  # Now everything is on same device
        return z, log_prob


