import pytest
import torch
import torch.nn as nn
from ..activation import *
from ..flatten import Flatten
from ..linear import Linear


def test_ReLU():
    X = torch.randn(10, 4)
    test = ReLU()
    real = nn.ReLU()
    assert torch.allclose(test(X), real(X))


def test_Softmax():
    X = torch.randn(10, 4)
    test = Softmax(dim=1)
    real = nn.Softmax(dim=1)
    assert torch.allclose(test(X), real(X))


def test_LogSoftmax():
    X = torch.randn(10, 4)
    test = LogSoftmax(dim=1)
    real = nn.LogSoftmax(dim=1)
    assert torch.allclose(test(X), real(X))


def test_Flatten():
    X = torch.randn(32, 1, 5, 5)

    # With default parameters
    test = Flatten()
    real = nn.Flatten()
    assert torch.allclose(test(X), real(X))

    # With non-default parameters
    test = Flatten(start_dim=0, end_dim=2)
    real = nn.Flatten(start_dim=0, end_dim=2)
    assert torch.allclose(test(X), real(X))


def test_Linear():
    X = torch.randn(128, 20)
    weight = nn.Parameter(torch.randn(30, 20))
    bias = nn.Parameter(torch.randn(30))
    test = Linear(in_features=20, out_features=30)
    real = nn.Linear(in_features=20, out_features=30)
    test.weight, test.bias = weight, bias
    real.weight, real.bias = weight, bias
    assert torch.allclose(test(X), real(X))


if __name__ == '__main__':
    pytest.main()
