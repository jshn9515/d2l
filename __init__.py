import os
import sys
import d2l.nn as nn
import d2l.optim as optim
import d2l.utils as utils
from packaging import version

__all__ = ['get_project_rootpath', 'get_project_dataset_path']

__author__ = 'jshn9515@163.com'
__version__ = '0.1.5'

# Check the Python version
python = version.parse(sys.version.split()[0])
if python < version.parse('3.10.0'):
    raise EnvironmentError(f'd2l package requires Python 3.10.0 or higher, but the current Python is {python}.')

try:
    import torch
except ImportError:
    raise ImportError('torch is not installed. Use `pip install torch` to install it.')


def get_project_rootpath():
    """ Get the root path of the project. """
    path = os.path.realpath(os.curdir)
    while True:
        if '.idea' or '.vscode' in os.listdir(path):
            return path
        path = os.path.dirname(path)


def get_project_dataset_path():
    """ Get the dataset path of the project. """
    path = get_project_rootpath()
    return os.path.join(path, 'datasets')
