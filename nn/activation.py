import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import d2l.nn.functional as d2l
from typing import Optional

__all__ = ['ReLU', 'Softmax', 'LogSoftmax', 'Sigmoid']


class ReLU(nn.Module):
    def __init__(self, fast: bool = False):
        """ Applies the rectified linear unit function. See :class:`torch.nn.ReLU` for more information. """
        super().__init__()
        self.fast = fast

    def forward(self, X: Tensor) -> Tensor:
        if self.fast:
            return F.relu(X)
        return torch.clamp_min(X, min=0)


class Softmax(nn.Module):
    def __init__(self, dim: Optional[int] = None, fast: bool = False):
        """ Applies the Softmax function. See :class:`torch.nn.Softmax` for more information. """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, X: Tensor) -> Tensor:
        if self.fast:
            return F.softmax(X, dim=self.dim)
        return d2l.softmax(X, dim=self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class LogSoftmax(nn.Module):
    def __init__(self, dim: Optional[int] = None, fast: bool = False):
        """ Applies the log_softmax function. See :class:`torch.nn.LogSoftmax` for more information. """
        super().__init__()
        self.dim = dim
        self.fast = fast

    def forward(self, X: Tensor) -> Tensor:
        if self.fast:
            return F.log_softmax(X, dim=self.dim)
        return d2l.log_softmax(X, dim=self.dim)

    def extra_repr(self):
        return f'dim={self.dim}'


class Sigmoid(nn.Module):
    """ Applies the sigmoid function. See :class:`torch.nn.Sigmoid` for more information. """
    def __init__(self, fast: bool = False):
        super().__init__()
        self.fast = fast

    def forward(self, X: Tensor) -> Tensor:
        if self.fast:
            return F.sigmoid(X)
        return d2l.sigmoid(X)
