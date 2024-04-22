from typing import Iterator
import abc
import torch.nn as nn

__all__ = ['SGD']


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def zero_grad(self):
        pass


class SGD(Optimizer):
    def __init__(self, params: Iterator[nn.Parameter], lr: float = 1e-3):
        super().__init__()
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
