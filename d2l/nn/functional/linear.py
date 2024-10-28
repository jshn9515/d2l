import torch
from torch import Tensor
from typing import Optional

__all__ = ['linear']


def linear(X: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """ Applies a linear transformation. See :func:`torch.nn.functional.linear` for more information. """
    if bias is None:
        return torch.matmul(X, weight.T)
    else:
        return torch.addmm(bias, X, weight.T)
