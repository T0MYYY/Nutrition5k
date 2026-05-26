from setuptools import setup, find_packages

setup(
    name="nutrition5k_pkg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "pandas",
        "PyYAML",
        "Pillow",
        "numpy",
    ],
)
