# Gassmann


## Overview

This repository provides an example implementation for inversion using different inference methods in a 5-dimensional model space. It leverages Markov chain Monte Carlo (McMC), stochastic Stein Variational Gradient Descent (sSVGD), Variational Inference with Normalising Flows and Simulation-Based Inference (SBI) to solve a simple inference problem. 
The forward model is one of the Gassmann equation and is defined in the src/utilities/Gassmann.py python file.


## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Using the code

To run the full-dimensional inference problem using McMC, sSVGD, Normalising Flows and SBI, please refer to the respective python files in the src/example/full folder
To run the marginalised inference problem, run the python files in the src/example/marg folder with respective Bayesian methods.


## License
This project is licensed under the Apache-2.0 License.

## References and Acknowledgements
This repository uses the following open-source software libraries:

numPy v2.3.1 (https://github.com/numpy/numpy)
sciPy v1.16.0 (https://github.com/scipy/scipy)
matplotlib v3.10.3 (https://github.com/matplotlib/matplotlib)
pytorch v2.5.1 (https://github.com/pytorch/pytorch)
normflows v1.7.3 (https://github.com/VincentStimper/normalizing-flows)
pints v0.5.0 (https://github.com/pints-team/pints)
sbi v0.25.0 (https://github.com/sbi-dev/sbi)

Sixtine Dromigny is funded by a UKRI NERC DTP Award (NE/S007474/1) and the Clarendon Scholarship (SR506) and gratefully acknowledges their support.

