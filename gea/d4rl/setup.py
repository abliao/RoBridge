from distutils.core import setup

from setuptools import find_packages

setup(
    name="d4rl",
    version="1.1",
    install_requires=[
        "gym",
        "numpy",
        "mujoco_py",
        "h5py",
        "termcolor",  # adept_envs dependency
        "click",  # adept_envs dependency
    ],
    packages=find_packages(),
)
