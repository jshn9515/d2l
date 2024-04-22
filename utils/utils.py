import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
from typing import Callable

__all__ = ['Accumulator', 'accuracy_1d', 'evaluate_accuracy_1d', 'accuracy_2d', 'evaluate_accuracy_2d',
           'rmse_loss_nd', 'log_rmse_loss_nd', 'evaluate_regression_nd']
logger = logging.getLogger(__name__)


class Accumulator:
    """ For accumulating sums over `n` variables. """

    def __init__(self, n: int):
        self.data = [0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0] * len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def is_matrix(X: torch.Tensor) -> bool:
    """ Check if a tensor is a matrix. """
    return X.ndim == 2


def accuracy_1d(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """ Compute the number of correct predictions. """
    if is_matrix(y_hat):  # y_hat is a matrix
        y_hat = torch.argmax(y_hat, dim=1)  # multi-class classification
    else:
        y_hat = torch.sigmoid(y_hat)  # binary classification
        y_hat = torch.where(y_hat >= 0.5, 1, 0)
    if is_matrix(y):  # y is a matrix
        y = torch.argmax(y, dim=1)
    y = y.type(y_hat.dtype)
    cmp = y_hat == y
    return float(torch.sum(cmp))


def evaluate_accuracy_1d(net: nn.Module, test: data.DataLoader) -> float:
    """ Compute the accuracy for a model on a dataset. The label must be the last column of DataLoader. """
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    with torch.no_grad():
        for *X, y in test:  # unpack for the first n inputs and the last input
            metric.add(accuracy_1d(net(*X), y), y.shape[0])
    return metric[0] / metric[1]


#  TODO: The behavior of this function is not complete
def accuracy_2d(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    """ Compute the number of correct predictions. """
    y_hat = torch.sigmoid(y_hat)
    y_hat = torch.where(y_hat >= 0.5, 1, 0)
    cmp = y_hat == y
    return float(torch.sum(cmp))


def evaluate_accuracy_2d(net: nn.Module, test: data.DataLoader) -> float:
    """ Compute the accuracy for a model on a dataset. The label must be the last column of DataLoader. """
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    with torch.no_grad():
        for *X, y in test:  # unpack for the first n inputs and the last input
            metric.add(accuracy_2d(net(*X), y), y.numel())
    return metric[0] / metric[1]


def rmse_loss_nd(y_hat: torch.Tensor, y: torch.Tensor):
    """ Compute the RMSE. """
    rmse = torch.sqrt(F.mse_loss(y_hat, y))
    return float(rmse)


def log_rmse_loss_nd(y_hat: torch.Tensor, y: torch.Tensor):
    """ Compute the log_RMSE. Do not standardize the label when using log_rmse. """
    loss = torch.sqrt(F.mse_loss(torch.log(y_hat), torch.log(y)))
    return float(loss)


def evaluate_regression_nd(net: nn.Module, test: data.DataLoader, func: Callable):
    """ Compute the loss for a model on a dataset. """
    metric = Accumulator(2)
    with torch.no_grad():
        for *X, y in test:
            metric.add(func(net(*X), y), 1)
    return metric[0] / metric[1]


def simple_classifier_train_1d(
        net: nn.Module,
        optimizer: optim.Optimizer,
        loss: nn.Module,
        train: data.DataLoader,
        validate: data.DataLoader = None,
        test: data.DataLoader = None,
        epochs: int = 10,
        verbose: bool = 1
):
    metrics = Accumulator(3)  # train_loss, train_accuracy, num_examples
    train_loss, train_accuracy, val_accuracy = [], [], []
    for epoch in range(epochs):
        net.train()
        metrics.reset()
        for *X, y in train:
            optimizer.zero_grad()
            l = loss(net(*X), y)
            l.backward()
            optimizer.step()
            metrics.add(float(l), accuracy_1d(net(X), y), y.shape[0])
        net.eval()
        train_loss.append(metrics[0] / metrics[2])
        train_accuracy.append(metrics[1] / metrics[2])
        if validate is not None:
            val_accuracy.append(evaluate_accuracy_1d(net, validate))
        if verbose:
            print(f'epoch {epoch + 1}, loss {train_loss[-1]:.4f}, train accuracy {train_accuracy[-1]:.4f},',
                  f'validation accuracy {val_accuracy[-1]:.4f}')
    if test is not None:
        net.eval()
        test_accuracy = evaluate_accuracy_1d(net, test)
        if verbose:
            print(f'test accuracy {test_accuracy:.4f}')


def simple_accuracy_plot(
        train_loss: list[float],
        train_accuracy: list[float],
        validate_accuracy: list[float] = None,
        epochs: int = 10
):
    plt.rcParams['font.size'] = 12
    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(range(1, epochs + 1), train_loss, color='tab:blue', label='training loss')
    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), train_accuracy, color='tab:orange', label='training accuracy')
    if validate_accuracy is not None:
        ax2.plot(range(1, epochs + 1), validate_accuracy, color='tab:green', label='validation accuracy')
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('training history')
    plt.show()
