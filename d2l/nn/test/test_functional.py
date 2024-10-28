import pytest
import torch
import d2l.nn.functional as d2l
import torch.nn.functional as F


def test_relu():
    X = torch.randn(30, 40)
    test = d2l.relu(X)
    real = F.relu(X)
    assert torch.allclose(test, real)


def test_softmax():
    X = torch.randn(30, 40)
    test = d2l.softmax(X, dim=1)
    real = F.softmax(X, dim=1)
    assert torch.allclose(test, real)


def test_log_softmax():
    X = torch.randn(30, 40)
    test = d2l.log_softmax(X, dim=1)
    real = F.log_softmax(X, dim=1)
    assert torch.allclose(test, real)


def test_linear():
    X = torch.randn(30, 40)
    weight = torch.randn(50, 40)
    bias = torch.randn(50)
    test = d2l.linear(X, weight, bias)
    real = F.linear(X, weight, bias)
    assert torch.allclose(test, real)


if __name__ == '__main__':
    pytest.main()
