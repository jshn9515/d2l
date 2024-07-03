import torch
from torch import Tensor
from typing import Optional

__all__ = ['relu', 'softmax', 'log_softmax', 'sigmoid', 'tanh']


def relu(X: Tensor) -> Tensor:
    """ Applies the rectified linear unit function. See :func:`torch.nn.functional.relu` for more information. """
    return torch.clamp_min(X, min=0)


def softmax(X: Tensor, dim: Optional[int] = None) -> Tensor:
    """ Applies the Softmax function. See :func:`torch.nn.functional.softmax` for more information. """
    exp = torch.exp(X)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def log_softmax(X: Tensor, dim: Optional[int] = None) -> Tensor:
    """ Applies the log_softmax function. See :func:`torch.nn.functional.log_softmax` for more information. """
    return torch.log(softmax(X, dim=dim))


def sigmoid(X: Tensor) -> Tensor:
    """ Applies the sigmoid function. See :func:`torch.nn.functional.sigmoid` for more information. """
    return 1 / (1 + torch.exp(-X))


def tanh(X: Tensor) -> Tensor:
    """ Applies the hyperbolic tangent function. See :func:`torch.nn.functional.tanh` for more information. """
    exp1 = torch.exp(X)
    exp2 = torch.exp(-X)
    return (exp1 - exp2) / (exp1 + exp2)
