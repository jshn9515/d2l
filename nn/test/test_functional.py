import pytest
import torch
import torch.nn.functional as F
from ..functional import *


def test_relu():
    X = torch.randn(30, 40)
    test = relu(X)
    real = F.relu(X)
    assert torch.allclose(test, real)


def test_softmax():
    X = torch.randn(30, 40)
    test = softmax(X, dim=1)
    real = F.softmax(X, dim=1)
    assert torch.allclose(test, real)


def test_log_softmax():
    X = torch.randn(30, 40)
    test = log_softmax(X, dim=1)
    real = F.log_softmax(X, dim=1)
    assert torch.allclose(test, real)


def test_linear():
    X = torch.randn(30, 40)
    weight = torch.randn(50, 40)
    bias = torch.randn(50)
    test = linear(X, weight, bias)
    real = F.linear(X, weight, bias)
    assert torch.allclose(test, real)


if __name__ == '__main__':
    pytest.main()
