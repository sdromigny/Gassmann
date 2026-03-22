import numpy as np
import torch
from geomloss import SamplesLoss
from ignite.engine import Engine
from ignite.metrics import MaximumMeanDiscrepancy
import pandas as pd


import matplotlib.pyplot as plt


mcmc_samples_det=np.load("./src/example/samples/marg/mcmc_samples_det.npy")

mcmc_samples_prob=np.load("./src/example/samples/marg/mcmc_samples_prob_pmh.npy")


nf_samples_det=np.load("./src/example/samples/marg/det/nf_samples_det.npy")

nf_samples_prob=np.load("./src/example/samples/marg/prob/nf_samples_prob.npy")

fmpe_samples_det=np.load("./src/example/samples/marg/det/fmpe_samples_det.npy")

fmpe_samples_prob=np.load("./src/example/samples/marg/prob/fmpe_samples_prob.npy")

ssvgd_samples_det=np.load("./src/example/samples/marg/det/ssvgd_samples_det.npy")

ssvgd_samples_prob=np.load("./src/example/samples/marg/prob/ssvgd_samples_prob_fin.npy")



#################################################################################
# Sinkhorn test

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
    std = values.std(ddof=1)  # unbiased std estimate

    return mean, std



device = "cuda" if torch.cuda.is_available() else "cpu"

def to_tensor(arr):
    return torch.from_numpy(arr).float().to(device)

mcmc_det = to_tensor(mcmc_samples_det)
mcmc_prob = to_tensor(mcmc_samples_prob)

methods = {
    # "NF (det)": to_tensor(nf_samples_det),
    # "FMPE (det)": to_tensor(fmpe_samples_det),
    "SSVGD (det)": to_tensor(ssvgd_samples_det),
    # "NF (prob)": to_tensor(nf_samples_prob),
    # "FMPE (prob)": to_tensor(fmpe_samples_prob),
    #"SSVGD (prob)": to_tensor(ssvgd_samples_prob),
}


sinkhorn_loss = SamplesLoss(
    "sinkhorn",
    p=2,
    blur=0.05,
    scaling=0.9,
    debias=False,
)

n_sub=100
n_boot=50

results = []

for name, samples in methods.items():
    print("new iteration")
    ref = mcmc_det if "(det)" in name else mcmc_prob

    # Sinkhorn
    sh_mean, sh_std = bootstrap_metric_std(
        ref,
        samples,
        compute_sinkhorn,
        n_sub=n_sub,
        n_boot=n_boot,
        loss_fn=sinkhorn_loss,
    )

    # MMD
    mmd_mean, mmd_std = bootstrap_metric_std(
        ref,
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

