import os
import argparse
import sys
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping


from utilities.TrainFlowMatching import TrainFlowMatch

from utilities.Gassmann import simulator_det, simulator_prob, simulator_full5


# Set up the argument parser and wandb logger project name
parser = argparse.ArgumentParser(description="Inversion with flow matching 5d")
parser.add_argument("--name", type=str, default="FMPEGass5_param", help="Project name for logging and saving results")


# --- Model space parameters ---
parser.add_argument("--input_dim", type=int, default=5, help="Dimensionality of the model space")
parser.add_argument("--output_dim", type=int, default=2, help="Dimensionality of the model space")

# --- Data Parameters ---
parser.add_argument("--sigma", type=float, default=0.01, help="Standard deviation for likelihood computation")
parser.add_argument("--d_obs", type=int, default=torch.tensor([0.64704126, 0.61732611]), help="Number of samples to plot")

# --- Training Parameters ---
parser.add_argument("--epochs", type=int, default=3000, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training or computation")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
parser.add_argument("--warmup", type=float, default=20, help="Warmup epochs for the learning rate scheduler")

# --- Plotting Parameters ---
parser.add_argument("--seed", type=int, default=32, help="Random seed for reproducibility")
parser.add_argument("--N", type=int, default=5000, help="Number of samples to plot")
parser.add_argument("--batch_plot", type=int, default=25, help="Batch size for plotting, depends on your device's memory")

# --- Forward model Selection ---
parser.add_argument(
    "--simulator", 
    type=str, 
    choices=["simulator_det", "simulator_prob", "simulator_full5"],  
    default="simulator_full5",  
    help="Choose the forward model: 'simulator_det' or 'simulator_prob'"
)

# --- Vector field Architecture Parameters ---
parser.add_argument("--hidden_units", type=int, default=5, help="Hidden untis for the model (determines width of the layers of the neural network)")
parser.add_argument("--embedding", type=int, default=0, help="embedding net for the model, default 0 means we use nn.Identity (no complex relationship between input and condition) if 1, we use a simple linear layer")
parser.add_argument("--noise_scale", type=float, default=0.0001, help="Noise scale for the vector field learning (epsilon)")
# Parse arguments
args = parser.parse_args()


# **Fix: Map simulator string to actual function**
simulator_map = {
    "simulator_det": simulator_det,
    "simulator_prob": simulator_prob,
    "simulator_full5": simulator_full5,
}
simulator = simulator_map[args.simulator]  # Now, `simulator` is a function, not a string


# **Initialize Weights & Biases (Wandb) Logger**
wandb_logger = WandbLogger(project=args.name, config=args)

# **Set the random seed for reproducibility**
seed_everything(int(args.seed))

# **Fix: Pass `nfm` into `TrainNormFlow`**
flow = TrainFlowMatch(args, wandb_logger, simulator=simulator)




# **Define the EarlyStopping callback** ------ Convergence criterium (stop training when the validation loss does not decrease anymore for a certain number of epochs defined inpatience
early_stopping = EarlyStopping(
    monitor="val_loss",      
    mode="min",              
    patience=1000,             
    min_delta=0,            
    verbose=True             
)

# **Fix: Create Trainer**
trainer = Trainer(
    max_epochs=args.epochs, 
    callbacks=[early_stopping], 
    logger=wandb_logger, 
    accelerator="cpu", 
    gradient_clip_val=1.0
)

# **Train the model**
trainer.fit(flow)

# **Plot Results**
fontsize = 25
m_true = torch.tensor([4, 7])
flow.plot(m_true, fontsize=fontsize)