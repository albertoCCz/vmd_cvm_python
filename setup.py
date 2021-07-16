from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules  = cythonize("pruebas.pyx"),
    include_dirs = [np.get_include()]
)
