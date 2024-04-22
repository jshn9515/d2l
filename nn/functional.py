import torch

__all__ = ['relu', 'softmax', 'log_softmax', 'linear']


def relu(X: torch.Tensor) -> torch.Tensor:
    """ Applies the rectified linear unit function. See :func:`torch.nn.functional.relu` for more information. """
    return torch.clamp_min(X, min=0)


def softmax(X: torch.Tensor, dim: int = None) -> torch.Tensor:
    """ Applies the Softmax function. See :func:`torch.nn.functional.softmax` for more information. """
    exp = torch.exp(X)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def log_softmax(X: torch.Tensor, dim: int = None) -> torch.Tensor:
    """ Applies the log_softmax function. See :func:`torch.nn.functional.log_softmax` for more information. """
    return torch.log(softmax(X, dim=dim))


def linear(X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """ Applies a linear transformation. See :func:`torch.nn.functional.linear` for more information. """
    if bias is None:
        return torch.matmul(X, weight.T)
    else:
        # another way to implement this is to use torch.addmm
        # return torch.addmm(self.bias, X, self.weight.T)
        return torch.matmul(X, weight.T) + bias
