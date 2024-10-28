import torch.optim as optim
from typing import Any, Iterator

__all__ = ['SGD']


class SGD(optim.Optimizer):
    def __init__(self, params: Iterator, lr: float = 0.01, momentum: float = 0, weight_decay: float = 0):
        if lr < 0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('differentiable', False)
            group.setdefault('fused', False)

    def step(self, closure: Any = None):
        if closure is not None:
            raise NotImplementedError('closure is not supported')
        for param_group in self.param_groups:
            for param in param_group['params']:
                param.data.add_(param.grad, alpha=-self.defaults['lr'])
