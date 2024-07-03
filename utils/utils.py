import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from typing import Callable

__all__ = ['Accumulator', 'accuracy_1d', 'evaluate_accuracy_1d', 'accuracy_2d', 'evaluate_accuracy_2d',
           'rmse_loss_nd', 'log_rmse_loss_nd', 'softmax_rmse_loss_nd', 'evaluate_rmse_loss_nd',
           'evaluate_log_rmse_loss_nd', 'evaluate_softmax_rmse_loss_nd']
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


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """ Saves model when validation loss decrease. """
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def is_matrix(X: Tensor) -> bool:
    """ Check if a tensor is a matrix. """
    return X.ndim == 2


def accuracy_1d(y_hat: Tensor, y: Tensor) -> float:
    """
    Compute the number of correct predictions, in one-dimensional.

    For CrossEntropyLoss, the label should be the class index. y_hat is always a 2D tensor, even in binary case,
    while y is a 1D torch.LongTensor. For BCEWithLogitsLoss, the shape of y_hat and y is always the same, and
    the dtype of y is torch.FloatTensor.

    Shape:
        - y_hat: for multi-class, (N, C); for binary class, (N, 1).
        - y: for multi-class, (N,); for binary class, (N, 1).
    """
    if y_hat.shape == y.shape and y.shape[1] > 1:  # probabilities
        raise ValueError('You are trying to compute accuracy on probabilities. Which is not allowed.')
    if y.dtype == torch.long:  # CrossEntropyLoss
        y_hat = torch.argmax(y_hat, dim=1)
    elif y.dtype == torch.float32:  # BCEWithLogitsLoss
        y_hat = torch.sigmoid(y_hat)
        y_hat = torch.where(y_hat >= 0.5, 1, 0)
    else:
        raise TypeError(f'Expected torch.LongTensor or torch.FloatTensor, got {type(y)}')
    y = y.type(y_hat.dtype)
    cmp = torch.isclose(y_hat, y)
    return float(torch.sum(cmp))


def evaluate_accuracy_1d(net: nn.Module, test: data.DataLoader) -> float:
    """ Compute the accuracy for a model on a dataset. The label must be the last column of DataLoader. """
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    with torch.no_grad():
        for *X, y in test:  # unpack for the first n inputs and the last input
            metric.add(accuracy_1d(net(*X), y), y.shape[0])
    return metric[0] / metric[1]


def accuracy_2d(y_hat: Tensor, y: Tensor) -> float:
    """
    Compute the number of correct predictions, in two-dimensional.

    Shape:
        - y_hat: for multi-class, (N, C, H, W); for binary class, (N, 1, H, W).
        - y: (N, H, W).
    """
    if y_hat.shape[1] > 1:  # multi-class
        y_hat = torch.argmax(y_hat, dim=1)
    else:  # binary class
        y_hat = torch.sigmoid(y_hat)
        y_hat = torch.where(y_hat >= 0.5, 1, 0)
    y = y.type(y_hat.dtype)
    cmp = torch.isclose(y_hat, y)
    return float(torch.sum(cmp))


def evaluate_accuracy_2d(net: nn.Module, test: data.DataLoader) -> float:
    """ Compute the accuracy for a model on a dataset. The label must be the last column of DataLoader. """
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    with torch.no_grad():
        for *X, y in test:  # unpack for the first n inputs and the last input
            metric.add(accuracy_2d(net(*X), y), y.numel())
    return metric[0] / metric[1]


def rmse_loss_nd(y_hat: Tensor, y: Tensor):
    """ Compute the RMSE. """
    rmse = torch.sqrt(F.mse_loss(y_hat, y))
    return float(rmse)


def log_rmse_loss_nd(y_hat: Tensor, y: Tensor):
    """ Compute the log_RMSE. Do not standardize the label when using log_rmse. """
    loss = torch.sqrt(F.mse_loss(torch.log(y_hat), torch.log(y)))
    return float(loss)


def softmax_rmse_loss_nd(y_hat: Tensor, y: Tensor):
    """ Compute the log_RMSE. Do not standardize the label when using log_rmse. """
    loss = torch.sqrt(F.mse_loss(F.softmax(y_hat, dim=1), y))
    return float(loss)


def evaluate_regression_nd(net: nn.Module, test: data.DataLoader, func: Callable):
    """ Compute the loss for a model on a dataset. """
    metric = Accumulator(2)
    with torch.no_grad():
        for *X, y in test:
            metric.add(func(net(*X), y), 1)
    return metric[0] / metric[1]


def evaluate_rmse_loss_nd(net: nn.Module, test: data.DataLoader):
    """ Compute the RMSE for a model on a dataset. """
    return evaluate_regression_nd(net, test, rmse_loss_nd)


def evaluate_log_rmse_loss_nd(net: nn.Module, test: data.DataLoader):
    """ Compute the log_RMSE for a model on a dataset. """
    return evaluate_regression_nd(net, test, log_rmse_loss_nd)


def evaluate_softmax_rmse_loss_nd(net: nn.Module, test: data.DataLoader):
    """ Compute the softmax RMSE for a model on a dataset. """
    metric = Accumulator(2)
    with torch.no_grad():
        for *X, y in test:
            metric.add(rmse_loss_nd(F.softmax(net(*X), dim=1), y), 1)
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
    import matplotlib.pyplot as plt
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
