"""Setup script for object_detection."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['Pillow>=1.0', 'Matplotlib>=2.1', 'Cython>=0.28.1']

setup(
    name='deeplab',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=['deeplab', 'deeplab.core', 'deeplab.datasets', 'deeplab.utils'],
    description='Tensorflow Deeplab Library',
)
