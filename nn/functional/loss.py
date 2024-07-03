import torch
import d2l.nn.functional as F
from torch import Tensor
from typing import Literal, Optional

__all__ = ['nll_loss', 'cross_entropy']
Reduction = Literal['mean', 'sum', 'none']


def nll_loss_1d(X: Tensor, y: Tensor, weight: Optional[Tensor] = None, reduction: Reduction = 'mean') -> Tensor:
    """ Compute the negative log likelihood loss for 1-dimensional target.
    Corresponding to C++ function: :func:`nll_loss_symint`. """
    X = torch.gather(X, dim=1, index=y.view(-1, 1))
    X = torch.squeeze(X)
    if weight is not None:
        match reduction:
            case 'mean':
                return -torch.sum(X * weight[y]) / torch.sum(weight[y])
            case 'sum':
                return -torch.sum(X * weight[y])
            case 'none':
                return -X * weight[y]
    else:
        match reduction:
            case 'mean':
                return -torch.sum(X) / X.shape[0]
            case 'sum':
                return -torch.sum(X)
            case 'none':
                return -X


def nll_loss_2d(X: Tensor, y: Tensor, weight: Optional[Tensor] = None, reduction: Reduction = 'mean') -> Tensor:
    """ Compute the negative log likelihood loss for 2-dimensional target.
    Corresponding to C++ function: :func:`nll_loss_2d_symint`. """
    target = y.view(-1, 1)
    X = torch.permute(X, dims=(0, 2, 3, 1))
    X = torch.reshape(X, (-1, X.shape[-1]))
    X = torch.gather(X, dim=1, index=target)
    if weight is not None:
        match reduction:
            case 'mean':
                return -torch.sum(X * weight[target]) / torch.sum(weight[target])
            case 'sum':
                return -torch.sum(X * weight[target])
            case 'none':
                return -torch.reshape(X * weight[target], y.shape)
    else:
        match reduction:
            case 'mean':
                return -torch.sum(X) / X.shape[0]
            case 'sum':
                return -torch.sum(X)
            case 'none':
                return -torch.reshape(X, y.shape)


def nll_loss(X: Tensor, y: Tensor, weight: Optional[Tensor] = None, reduction: Reduction = 'mean') -> Tensor:
    """ Compute the negative log likelihood loss for n-dimensional input.
    Corresponding to C++ function: :func:`nll_loss_nd_symint`. """
    class_dim = 0 if X.ndim == 1 else 1
    n_class = X.shape[class_dim]
    if weight is not None and (weight.ndim != 1 or weight.numel() != n_class):
        raise ValueError(f'nll_loss: weight tensor should be defined either for all {n_class},'
                         f' classes or no classes but got weight tensor of shape: {weight.shape}')
    if X.ndim == 2:
        return nll_loss_1d(X, y, weight=weight, reduction=reduction)
    elif X.ndim >= 4:  # [N, C, W, H]
        return nll_loss_2d(X, y, weight=weight, reduction=reduction)
    else:
        raise NotImplementedError(f'Unsupported input dimension: {X.ndim}')


def cross_entropy_loss_probability(X: Tensor, y: Tensor, weight: Optional[Tensor] = None, reduction: Reduction = 'mean') -> Tensor:
    """ Compute the cross entropy loss probabilities between predicted probabilities and target.
    Corresponding to C++ function: :func:`cross_entropy_loss_prob_target`. """
    class_dim = 0 if X.ndim == 1 else 1
    n_class = X.shape[class_dim]
    if weight is not None and (weight.ndim != 1 or weight.numel() != n_class):
        raise ValueError(f'cross_entropy: weight tensor should be defined either for all {n_class}, '
                         f'classes or no classes but got weight tensor of shape: {weight.shape}')
    X = F.log_softmax(X, dim=class_dim)
    if weight is not None:
        match reduction:
            case 'mean':
                return -torch.sum(X * y * weight) / (X.numel() // n_class)
            case 'sum':
                return -torch.sum(X * y * weight)
            case 'none':
                return -torch.sum(X * y * weight, dim=class_dim)
    else:
        match reduction:
            case 'mean':
                return -torch.sum(X * y) / (X.numel() // n_class)
            case 'sum':
                return -torch.sum(X * y)
            case 'none':
                return -torch.sum(X * y, dim=class_dim)


def cross_entropy(X: Tensor, y: Tensor, weight: Optional[Tensor] = None, reduction: Reduction = 'mean') -> Tensor:
    """ Computes the cross entropy loss between input logits and target. Corresponding to C++ function:
     :func:`cross_entropy_loss_symint`. Note: ``label_smoothing`` is not implemented in this function. """
    class_dim = 0 if X.ndim == 1 else 1
    if X.shape == y.shape:  # probabilities
        if y.dtype != torch.float32:
            raise TypeError(f'Expected floating point type for target with class probabilities, got {y.dtype}')
        ret = cross_entropy_loss_probability(X, y, weight=weight, reduction=reduction)
    else:  # indices
        ret = nll_loss(F.log_softmax(X, dim=class_dim), y, weight=weight, reduction=reduction)
    return ret
