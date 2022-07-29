from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext = [Extension(
    name="cdfcalc",
    sources=["vmd_cvm_cython/CDFCALC.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
  ),
  Extension(
    name="cvm",
    sources=["vmd_cvm_cython/CVM.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
  ),
  Extension(
    name="ecdf",
    sources=["vmd_cvm_cython/ecdf.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
  ),
  Extension(
    name="Prop_VMD_CVM",
    sources=["vmd_cvm_cython/Prop_VMD_CVM.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
  ),
  Extension(
    name="threshvspfa",
    sources=["vmd_cvm_cython/THRESHVSPFA.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
  ),
]
setup(
    ext_modules  = cythonize(ext, language_level=3, annotate=False),
)
