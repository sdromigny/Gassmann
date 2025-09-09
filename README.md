# Gassmann

## Overview

This repository provides an example implementation for inversion using flow matching in a 5-dimensional model space. It leverages PyTorch Lightning, Wandb for experiment tracking, and custom simulators for forward modeling.  
**(Please add a more detailed description here.)**

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Usage Examples

Below are three example commands to run the main script with different flags.  
**(Please update these examples as needed.)**

```bash
# Example 1: Run with default settings
python src/example/full/FMPE.py

# Example 2: Specify a different simulator and project name
python src/example/full/FMPE.py --simulator simulator_det --name MyProject

# Example 3: Change training parameters
python src/example/full/FMPE.py --epochs 5000 --batch_size 64 --lr 0.0005
```

Refer to the script’s help for all available flags:

```bash
python src/example/full/FMPE.py --help
```