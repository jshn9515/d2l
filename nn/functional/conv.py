import torch
from torch import Tensor

__all__ = ['conv1d']
Type = int | tuple[int, ...]


def conv1d(X: Tensor, weight: Tensor, bias: Tensor = None, stride: Type = 1, padding: Type = 0, dilation: Type = 1, groups: Type = 1) -> Tensor:
    """ Applies a 1D convolution over an input signal composed of several input planes.
    See :func:`torch.nn.functional.conv1d` for more information.
    """
    return torch.conv1d(X, weight, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
