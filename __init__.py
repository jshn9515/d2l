import os
import d2l.nn as nn
import d2l.utils as utils

__all__ = ['get_project_rootpath', 'get_project_dataset_path']

__version__ = '0.1.0'


def get_project_rootpath():
    """ Get the root path of the project. """
    path = os.path.realpath(os.curdir)
    while True:
        if '.idea' in os.listdir(path):
            return path
        path = os.path.dirname(path)


def get_project_dataset_path():
    """ Get the dataset path of the project. """
    path = get_project_rootpath()
    return os.path.join(path, 'datasets')
