import torch.nn as nn
from torch import Tensor
import torch

class SimpleVectorFieldNet(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, time_encoding_dim: int, hidden_dim: int = 128):
        super().__init__()
        total_in = input_dim + condition_dim + time_encoding_dim
        self.net = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, theta: Tensor, x: Tensor, t: Tensor) -> Tensor:
        input_cat = torch.cat((theta, x, t), dim=-1)
        return self.net(input_cat)
