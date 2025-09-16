import numpy as np
from geomloss import SamplesLoss
import torch
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.clustering import *
from ignite.metrics.regression import *
from ignite.utils import *


import matplotlib.pyplot as plt


mcmc_samples_det=np.load("src/example/samples/mcmc_samples_det.npy")

mcmc_samples_prob=np.load("src/example/samples/samples/mcmc_samples_prob.npy")


nf_samples_det=np.load("src/example/samples/nf_samples_det.npy")

nf_samples_prob=np.load("src/example/samples/nf_samples_prob.npy")

fmpe_samples_det=np.load("src/example/samples/fmpe_samples_det.npy")

fmpe_samples_prob=np.load("src/example/samples/fmpe_samples_prob.npy")

ssvgd_samples_det=np.load("src/example/samples/ssvgd_samples_det.npy")

ssvgd_samples_prob=np.load("src/example/samples/ssvgd_samples_prob.npy")


#################################################################################
# Sinkhorn test



x = torch.from_numpy(mcmc_samples_prob).float()  # shape [N, d]
y = torch.from_numpy(ssvgd_samples_prob).float()    # shape [M, d]

# Optionally move to GPU if available:
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
y = y.to(device)

# Now create SamplesLoss once:
loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.9, debias=True)

# Compute loss:
sinkhorn_test = loss_fn(x[-1000:], y[-1000:])  # should no longer NameError
print("Sinkhorn distance:", sinkhorn_test)

########################################################

# Maximum Mean Discrepancy

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

x = torch.from_numpy(mcmc_samples_det).float()  # shape [N, d]
y = torch.from_numpy(ssvgd_samples_det).float()    # shape [M, d]

# Optionally move to GPU if available:
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)
y = y.to(device)

metric = MaximumMeanDiscrepancy()
metric.attach(default_evaluator, "mmd")

state = default_evaluator.run([[x[-1000:], y[-1000:]]])
print(state.metrics["mmd"])


