from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import d2l.nn.functional as d2l
from typing import Literal, Optional

__all__ = ['NLLLoss', 'CrossEntropyLoss']
Reduction = Literal['mean', 'sum', 'none']


class NLLLoss(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, reduction: Reduction = 'mean', fast: bool = False):
        """ This criterion computes the negative log likelihood loss between input logits and target. """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.fast = fast

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        if self.fast:
            return F.nll_loss(X, y, weight=self.weight, reduction=self.reduction)
        return d2l.nll_loss(X, y, weight=self.weight, reduction=self.reduction)


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: Optional[Tensor] = None, reduction: Reduction = 'mean', fast: bool = False):
        """ This criterion computes the cross entropy loss between input logits and target. """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.fast = fast

    def forward(self, X: Tensor, y: Tensor) -> Tensor:
        if self.fast:
            return F.cross_entropy(X, y, weight=self.weight, reduction=self.reduction)
        return d2l.cross_entropy(X, y, weight=self.weight, reduction=self.reduction)
