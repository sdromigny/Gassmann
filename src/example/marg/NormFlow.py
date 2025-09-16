import torch
import os
import sys

current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
import math
import normflows as nf
from tqdm import tqdm
from utilities.Gassmann import simulator_prob, simulator_det, sample_and_log_gaussians
from utilities.NormFlows import *

#from utils.TrainNormFlow import TrainNormFlow  # Adjust import paths as needed
from utilities.Histogram2d import pairplot
import os

import matplotlib.pyplot as plt



# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Flow model setup
K = 1
dim = 2
flows = [nf.flows.Planar((dim,)) for _ in range(K)]
q0 = nf.distributions.DiagGaussian(dim)
nfm = Linear(dim=dim, kernel='fullrank').to(device)


# Training config
num_epochs = 2000
batch_size = 50
sigma = 0.01
observed_data = torch.tensor([0.64704126, 0.61732611], device=device)

simulator=simulator_prob

loss_hist = np.array([])
best_loss = float('inf')
best_model_state = None



optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-3)

for it in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    z0 = torch.randn(batch_size, dim, device=device)

    w, log_det_flow = nfm.forward_and_log_det(z0)

    theta, log_det_trans = log_det_transform(w, low=0.0, high=10.0)

    log_det = log_det_flow + log_det_trans

    synthetic_value,n = simulator(theta.to(device))  # Ensure simulator returns device-correct outputs

    log_p = -0.5 * (((synthetic_value - observed_data) / sigma) ** 2).sum()

    log_q0 = -0.5 * torch.sum(z0**2, dim=1) - 0.5*dim*math.log(2*math.pi)
    log_q = log_q0 - log_det

    log_pn=sample_and_log_gaussians(n)
    loss = -(log_q + log_p + log_pn).mean()

    loss.backward()
    optimizer.step()

    loss_hist = np.append(loss_hist, loss.item())

    if loss < best_loss:
        best_loss = loss
        best_model_state = nfm.state_dict()
        print(f"Model saved at iteration {it} with loss: {best_loss}")

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, label='loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss for VI Normalising Flows")

save_path = "src/example/marg/results/nf_prob.png"


# Sampling and plotting
z, _ = nfm.sample(num_samples=2 ** 20)
a, b = 0.0, 10.0   # or your actual domain bounds
theta_samples = unconstrained_to_constrained(z, a, b)  # shape (N, dim)
m_true = torch.tensor([4, 7])

np.save("src/example/samples/nf_samples.npy",z.detach().cpu().numpy())

pairplot(theta_samples.detach().cpu().numpy(), m_true.detach().cpu().numpy(), fontsize=15, save_path=save_path)

##########################################################################################################################


# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Flow model setup
K = 1
dim = 2
flows = [nf.flows.Planar((dim,)) for _ in range(K)]
q0 = nf.distributions.DiagGaussian(dim)
nfm = Linear(dim=dim, kernel='fullrank').to(device)


# Training config
num_epochs = 1000
batch_size = 50
sigma = 0.01
observed_data = torch.tensor([0.64704126, 0.61732611], device=device)
#simulator = simulator_prob  # or simulator_det
simulator=simulator_det

loss_hist = np.array([])
best_loss = float('inf')
best_model_state = None



optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-3, weight_decay=1e-3)

for it in tqdm(range(num_epochs)):
    optimizer.zero_grad()

    z0 = torch.randn(batch_size, dim, device=device)
    # 2. Flow forward: w, log_det_flow = nfm.forward_and_log_det(z0)
    w, log_det_flow = nfm.forward_and_log_det(z0)

    theta, log_det_trans = log_det_transform(w, low=0.0, high=10.0)

    log_det = log_det_flow + log_det_trans

    synthetic_value = simulator(theta.to(device))  # Ensure simulator returns device-correct outputs

    log_p = -0.5 * (((synthetic_value - observed_data) / sigma) ** 2).sum()

    log_q0 = -0.5 * torch.sum(z0**2, dim=1) - 0.5*dim*math.log(2*math.pi)
    log_q = log_q0 - log_det

    loss = -(log_q + log_p).mean()

    loss.backward()
    optimizer.step()

    loss_hist = np.append(loss_hist, loss.item())

    if loss < best_loss:
        best_loss = loss
        best_model_state = nfm.state_dict()
        print(f"Model saved at iteration {it} with loss: {best_loss}")

# Plot loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, label='loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss for VI Normalising Flows")

save_path = "src/example/marg/results/nf_det.png"


# Sampling and plotting
z, _ = nfm.sample(num_samples=2 ** 20)
a, b = 0.0, 10.0   # or your actual domain bounds
theta_samples = unconstrained_to_constrained(z, a, b)  # shape (N, dim)
m_true = torch.tensor([4, 7])

np.save("src/example/samples/nf_samples.npy",z.detach().cpu().numpy())

pairplot(theta_samples.detach().cpu().numpy(), m_true.detach().cpu().numpy(), fontsize=15, save_path=save_path)
