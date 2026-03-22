
# Gassmann

A compact example repository that demonstrates several Bayesian inference approaches for a toy geophysical inverse problem based on a Gassmann equation as forward function.

---

## Overview

This repository provides working examples for solving a 5‑dimensional inversion problem using multiple inference methods:

* Markov chain Monte Carlo (McMC)
* Stochastic Stein Variational Gradient Descent (sSVGD)
* Variational Inference with Normalising Flows
* Simulation‑Based Inference (SBI) with Neural Density Estimation (NDE)

The forward function is the Gassmann equation (Gassmann, 1951). The implementation lives in `src/utilities/Gassmann.py` and provides three simulator entry points:

* `simulator_det`: deterministic simulator where nuisance parameters are fixed to their MLE values.
* `simulator_prob`: probabilistic simulator that samples nuisance parameters from their prior distributions.
* `simulator_full5`: treats all parameters (including previously nuisance parameters) as target variables for full 5‑dimensional inference.

Use the *full* examples to solve the full‑dimension problem, or the *marginal* examples to run the marginalised inference experiments.

---

## Features

* Example implementations of four inference strategies for a 5‑D Gassmann forward model.
* Scripts for both full‑dimensional and marginalised inference setups.
* Tools to reproduce a rejection‑sampling reweighting scheme for MCMC samples.
* Results and figures saved to the `results/` folder.

---

## Repository layout (important files & folders)

```
src/
  utilities/
    Gassmann.py            # forward functions
    MCMCFunc.py            # McMC sampling functions
    SVGDFunc.py            # SVGD and sSVGD sampling functions
    Histogram2d.py         # Plotting function for the marginalised inferences
    KL_example.py          # Reproduce simple Gaussian example to demonstrate the Jensen gap
    PlotHighD.py           # Plotting function for the full-dimensional inferences
    FlowMatchingEstimator.py, MLP.py          # Functions to run an alternative SBI algorithm, not based on Normalising Flows, but on Flow matching, which showed good results on the marginalised inference problem 
  example/
    full/                # scripts for full‑dimensional inference (MCMC, sSVGD, NF, SBI)
        results/         # where figures are saved
    marg/                # scripts for marginalised inference experiments (probabilistic and deterministic simulator)
        results/         # where figures are saved
    marg/RejectionSampling.py  # reweight MCMC samples using two weighting schemes
    samples/                 # where posterior samples are saved
requirements.txt         # Python package dependencies
README.md                # this file
```

---

## Installation

Create a virtual environment (recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

The repository was developed with:

* NumPy v2.3.1
* SciPy v1.16.0
* Matplotlib v3.10.3
* PyTorch v2.5.1
* normflows v1.7.3
* pints v0.5.0
* sbi v0.25.0

(See `requirements.txt` for the exhaustive list)

---

## Usage

### Full‑dimensional inference

Run the example scripts in `src/example/full/` to run the full‑dimensional (5‑D) inversion problem with the available inference methods. Example invocation pattern:

```bash
python src/example/full/<method_script>.py
```

Replace `<method_script>.py` with the script corresponding to the method you want to run (e.g. MCMC, sSVGD, NormFlows, or SBI). Each script will produce posterior distribution plots and save output to the `results/` folder and `samples/`.

### Marginalised inference

To run the marginalised inference experiments (where nuisance parameters are integrated out or treated separately), run the scripts in `src/example/marg/`:

```bash
python src/example/marg/<script>.py
```

The forward model call `simulator_det` corresponds to the case where the nuisance parameters are set to their Maximum Likelihood Estimates (MLEs) values. The forward model call `simulator_prob` corresponds to the case where we sample the nuisance parameters using the `sample_and_log_gaussians` function in `src/utilities/Gassmann.py`.

### Rejection sampling (reweighting MCMC samples)

To reproduce the rejection‑sampling reweighting of MCMC samples according to two different weighting schemes, run:

```bash
python src/example/marg/RejectionSampling.py
```

Provide or point the script to the MCMC samples you previously drew and saved to the `samples/` folder; the script will output reweighted sample sets and figures in `results/`.

### Comparison of the posterior estimates quality
 
Run the comparison script to compute two distance metrics between posterior samples produced by the different inference methods (Sinkhorn distance and Maximum Mean Discrepancy — MMD). From either the `src/example/full` or `src/example/marg` folder execute:

```bash
python src/example/marg/Comparison.py
```

The script loads saved sample arrays (MCMC, NF, SBI, sSVGD) from `src/example/samples/`. It then computes the Sinkhorn distance and MMD between the two posterior sample sets you select in the file, and prints the results to stdout. By default the script uses the last 1,000 samples for the Sinkhorn calculation and the last 10,000 samples for the MMD. To compare a different pair of posteriors, change the x and y assignments in the script (e.g. replace mcmc_samples / sbi_samples with any other pair), or change the slicing of the samples to compute the distances with.

---


## Forward model reference

Gassmann, F., 1951. *Elastic waves through a packing of spheres*, Geophysics, 16(4), 673–685.

The forward model code is implemented in `src/utilities/Gassmann.py` and documents the mapping between model parameters and the predicted observations used by the example inference pipelines.

---

# Bayesian inference methods references

Clerx, M., Robinson, M., Lambert, B., Lei, C. L., Ghosh, S., Mirams, G. R., & Gavaghan, D. J., 2018.
*Probabilistic inference on noisy time series (pints )*, arXiv preprint arXiv:1812.07388.
Stimper, V., Liu, D., Campbell, A., Berenz, V., Ryll, L., Sch¨olkopf, B., & Hern´andez-Lobato, J. M., 2023.
*normflows: A pytorch package for normalizing flows*, arXiv preprint arXiv:2302.12014.
Tejero-Cantero, A., Boelts, J., Deistler, M., Lueckmann, J.-M., Durkan, C., Gonc¸alves, P. J., Greenberg,
D. S., & Macke, J. H., 2020. *sbi: A toolkit for simulation-based inference*, Journal of Open Source
Software, 5(52), 2505. doi:10.21105/joss.02505.
Zhang, X., & Curtis, A. (2024). *VIP-Variational Inversion Package with example implementations of Bayesian tomographic imaging*. Seismica, 3(1).


## License

This project is licensed under the Apache‑2.0 License.

---

## Acknowledgements

Sixtine Dromigny is funded by a UKRI NERC DTP Award (NE/S007474/1) and the Clarendon Scholarship (SR506). The author gratefully acknowledges their support.

---

## Contact / Questions

If you have questions or want help reproducing a particular experiment, please open an issue or contact the repository author (sixtine.dromigny@stx.ox.ac.uk).
