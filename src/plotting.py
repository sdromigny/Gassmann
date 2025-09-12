import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import torch
from utilities.Histogram2d import pairplot
from utilities.PlotHighD import *

samples=np.load("/home/users/scro4690/Documents/GenInv/Gassmann/src/example/samples/full/mcmc_samples.npy")

save_path = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/full/results/mcmctest.png"

m_true=torch.tensor([4, 7])

plot_5d_corner(samples, save_path=save_path)