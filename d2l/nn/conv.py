import itertools
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import d2l.nn.functional as d2l
from torch.nn.modules.conv import _ConvNd
from typing import Any, Iterable

__all__ = ['Conv2d']


def repeat(x: Any, n: int):
    if isinstance(x, Iterable):
        return tuple(x)
    return tuple(itertools.repeat(x, n))


class Conv2d(_ConvNd):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int | tuple[int],
            stride: int | tuple[int] = 1,
            padding: str | int | tuple[int] = 0,
            dilation: int | tuple[int] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            fast: bool = False
    ):
        kernel_size = repeat(kernel_size, 1)
        stride = repeat(stride, 1)
        padding = padding if isinstance(padding, str) else repeat(padding, 1)
        dilation = repeat(dilation, 1)
        self.fast = fast
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            transposed=False,
            output_padding=repeat(0, 1),
            groups=groups,
            bias=bias,
            padding_mode=padding_mode
        )

    def forward(self, X: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        if self.fast:
            if self.padding_mode != 'zeros':
                X = F.pad(X, self._reversed_padding_repeated_twice, mode=self.padding_mode)
                return F.conv1d(X, weight, bias, self.stride, repeat(0, 1), self.dilation, self.groups)
            return F.conv1d(X, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return d2l.conv1d(X, weight, bias, self.stride, self.padding, self.dilation, self.groups)
