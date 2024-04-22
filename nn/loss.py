import torch
import torch.nn as nn
import torch.nn.functional as F
import d2l.nn.functional as d2l
from typing import Literal

__all__ = ['CrossEntropyLoss']
Reduction = Literal['mean', 'sum', 'none']


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor = None, reduction: Reduction = 'mean', fast: bool = False):
        """This criterion computes the cross entropy loss between input logits and target."""
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.fast = fast
        self.class_dim = None
        self.n_class = None

    def forward(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.fast:
            return F.cross_entropy(X, y, weight=self.weight, reduction=self.reduction)
        else:
            self.class_dim = 0 if X.ndim == 1 else 1
            return self.cross_entropy_loss_probability(X, y)

    def cross_entropy_loss_probability(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the cross entropy loss between predicted probabilities and target."""
        X = d2l.log_softmax(X, dim=self.class_dim)
        if self.weight:
            match self.reduction:
                case 'mean':
                    return -torch.sum(X * y * self.weight) / X.shape[0]
                case 'sum':
                    return -torch.sum(X * y * self.weight)
                case 'none':
                    return -torch.sum(X * y * self.weight, dim=self.class_dim)
        else:
            match self.reduction:
                case 'mean':
                    return -torch.sum(X * y) / X.shape[0]
                case 'sum':
                    return -torch.sum(X * y)
                case 'none':
                    return -torch.sum(X * y, dim=self.class_dim)
