import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import d2l.nn.functional as d2l

__all__ = ['Identity', 'Linear']


class Identity(nn.Module):
    """ A placeholder identity operator that is argument-insensitive. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X: Tensor) -> Tensor:
        return X


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, fast: bool = False):
        """ Applies a linear transformation to the incoming data. See :class:`torch.nn.Linear` for more information. """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fast = fast
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, X: Tensor) -> Tensor:
        if self.fast:
            return F.linear(X, self.weight, self.bias)
        return d2l.linear(X, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
