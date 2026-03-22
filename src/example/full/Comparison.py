import numpy as np
from geomloss import SamplesLoss
import torch
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.metrics.clustering import *
from ignite.metrics.regression import *
from ignite.utils import *
# from tarp import get_tarp_coverage
import pandas as pd
import matplotlib.pyplot as plt


mcmc_samples=np.load("./src/example/samples/full/mcmc_samples.npy")


nf_samples=np.load("./src/example/samples/full/nf_samples.npy")

sbi_samples=np.load("./src/example/samples/full/sbi.npy")

ssvgd_samples=np.load("./src/example/samples/full/ssvgd.npy")
ssvgd_samples=ssvgd_samples[-100000:]


def compute_sinkhorn(x, y, loss_fn, n_sub):
    idx_x = torch.randperm(x.shape[0])[:n_sub]
    idx_y = torch.randperm(y.shape[0])[:n_sub]
    return loss_fn(x[idx_x], y[idx_y]).item()


def compute_mmd(x, y, n_sub):
    idx_x = torch.randperm(x.shape[0])[:n_sub]
    idx_y = torch.randperm(y.shape[0])[:n_sub]

    def eval_step(engine, batch):
        return batch

    evaluator = Engine(eval_step)
    metric = MaximumMeanDiscrepancy()
    metric.attach(evaluator, "mmd")

    state = evaluator.run([[x[idx_x], y[idx_y]]])
    return state.metrics["mmd"]


def bootstrap_metric_std(
    x,
    y,
    metric_fn,
    n_sub=1000,
    n_boot=50,
    **metric_kwargs,
):
    values = []

    for _ in range(n_boot):
        val = metric_fn(x, y, n_sub=n_sub, **metric_kwargs)
        values.append(val)

    values = np.array(values)
    mean = values.mean()
    std = values.std(ddof=1)

    return mean, std


device = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(arr):
    return torch.from_numpy(arr).float().to(device)

mcmc = to_tensor(mcmc_samples)

methods = {
    # "NF": to_tensor(nf_samples),
    # "SBI": to_tensor(sbi_samples),
    "SSVGD": to_tensor(ssvgd_samples),
}

sinkhorn_loss = SamplesLoss(
    "sinkhorn",
    p=2,
    blur=0.05,
    scaling=0.9,
    debias=True,
)


n_sub = 1000
n_boot = 50

results = []

for name, samples in methods.items():
    print(f"Running {name}")

    sh_mean, sh_std = bootstrap_metric_std(
        mcmc,
        samples,
        compute_sinkhorn,
        n_sub=n_sub,
        n_boot=n_boot,
        loss_fn=sinkhorn_loss,
    )

    mmd_mean, mmd_std = bootstrap_metric_std(
        mcmc,
        samples,
        compute_mmd,
        n_sub=n_sub,
        n_boot=n_boot,
    )

    results.append(
        {
            "Method": name,
            "Sinkhorn": f"{sh_mean:.4f} ± {sh_std:.4f}",
            "MMD": f"{mmd_mean:.4f} ± {mmd_std:.4f}",
        }
    )


df = pd.DataFrame(results)
print(df.to_string(index=False))
