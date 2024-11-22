import setuptools
from d2l import __version__


setuptools.setup(
    name='d2l',
    version=__version__,
    description='A lightweight deep learning library',
    author='jshn9515',
    author_email='jshn9515@163.com',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'matplotlib', 'torch'],
)
