from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules  = cythonize("vmd_cvm_cython/*.pyx", language_level=3),
    include_dirs = [np.get_include()]
)
