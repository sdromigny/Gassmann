import torch
import os
import sys

# Add the 'src/' directory to Python path
current_dir = os.path.dirname(__file__)

src_path = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(src_path)

import numpy as np
import torch
from utilities.SVGDFunc import SVGDGassmannDet, SVGDGassmannProb, sSVGDGassmannProb, sSVGDGassmannDet  # the class file you wrote above
from utilities.Histogram2d import pairplot

samples=np.load("/home/users/scro4690/Documents/GenInv/SBIcompare/src/examples/gassmann/samples/filtered_mc.npy")

save_path = "/home/users/scro4690/Documents/GenInv/Gassmann/src/example/marg/results/reject.png"

m_true=torch.tensor([4, 7])


np.save("/home/users/scro4690/Documents/GenInv/Gassmann/src/example/samples/marg/reject.npy",samples)

pairplot(samples, m_true.detach().numpy(), fontsize=15, density_threshold=0,save_path=save_path)