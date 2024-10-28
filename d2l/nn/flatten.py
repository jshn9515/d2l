import torch
from torch import Tensor
import torch.nn as nn

__all__ = ['Flatten']


class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        """ Flattens a contiguous range of dims into a tensor. """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, X: Tensor) -> Tensor:
        return torch.flatten(X, self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return f'start_dim={self.start_dim}, end_dim={self.end_dim}'
