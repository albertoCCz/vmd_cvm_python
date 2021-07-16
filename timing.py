import timeit
import numpy as np

"""CVM
import_cy_module = "from CVM import cvm"
import_py_module = "from CVM_python import cvm as cvm_python"

cy_cvm = "cvm(15, 100)"
py_cvm = "cvm_python(15, 100)"

print("Cython", timeit.timeit(stmt=cy_cvm, setup=import_cy_module, number=100))
print("Python", timeit.timeit(stmt=py_cvm, setup=import_py_module, number=100))
"""


"""SNR
import_cy_module = "from SNR import snr"
import_py_module = "from SNR_python import snr as snr_python"

x0 = str(np.arange(0, 100, 1).tolist())
y0 = str(np.random.randint(0, 100, size=100).tolist())

cy_snr = f"snr({x0}, {y0})"
py_snr = f"snr_python({x0}, {y0})"

print("Cython", timeit.timeit(stmt=cy_snr, setup=import_cy_module, number=100))
print("Python", timeit.timeit(stmt=py_snr, setup=import_py_module, number=100))
"""


"""MSE
import_cy_module = "from MSE import mse"
import_py_module = "from MSE_python import mse as mse_python"

x0 = str(np.arange(0, 100, 1).tolist())
y0 = str(np.random.randint(0, 100, size=100).tolist())

cy_mse = f"mse({x0}, {y0})"
py_mse = f"mse_python({x0}, {y0})"

print("Cython", timeit.timeit(stmt=cy_mse, setup=import_cy_module, number=1000))
print("Python", timeit.timeit(stmt=py_mse, setup=import_py_module, number=1000))
"""


"""CDFCALC
import_cy_module = "from CDFCALC import cdfcalc"
import_py_module = "from CDFCALC_python import cdfcalc as cdfcalc_python"

x0   = np.random.randint(low=0, high=100, size=100).tolist()
disn = 7
ind  = 4

cy_cdf = f"cdfcalc({x0}, {disn}, {ind})"
py_cdf = f"cdfcalc_python({x0}, {disn}, {ind})"

print("Cython", timeit.timeit(stmt=cy_cdf, setup=import_cy_module, number=10000))
print("Python", timeit.timeit(stmt=py_cdf, setup=import_py_module, number=10000))
"""

"""ecdf
import_cy_module = "from ecdf import ecdf"
import_py_module = "from ecdf_python import ecdf as ecdf_python"

data = np.random.randint(low=0, high=100, size=50).tolist()

cy_cdf = f"ecdf({data})"
py_cdf = f"ecdf_python({data})"

print("Cython", timeit.timeit(stmt=cy_cdf, setup=import_cy_module, number=10000))
print("Python", timeit.timeit(stmt=py_cdf, setup=import_py_module, number=10000))

"""

import xarray as xr
import numpy as np

from VMD import vmd
from VMD_python import vmd as vmd_python

import time

# Some sample parameters for VMD
f = xr.load_dataset('Data/2015_Granada_Noisy.nc').beta_mean.values[0]    # Noisy signal
alpha = 2000      # moderate badwidth constraint
tau   = 0         # noise-tolerance (no strict fidelity enforcement)
NIMF  = 10        # IMFs to consider
DC    = 0         # no DC part imposed
init  = 1         # initialize omegas uniformly
tol   = 1e-7

# Timing Cython
start_cy = time.time()
x, y, z = vmd(f, alpha, tau, NIMF, DC, init, tol)
end_cy = time.time()

# Timing Python
start_py = time.time()
x_py, y_py, z_py = vmd_python(f, alpha, tau, NIMF, DC, init, tol)
end_py = time.time()

# text = """
# # Some sample parameters for VMD
# f = xr.load_dataset('Data/2015_Granada_Noisy.nc').beta_mean.values[0]    # Noisy signal
# alpha = 2000      # moderate badwidth constraint
# tau   = 0         # noise-tolerance (no strict fidelity enforcement)
# NIMF  = 10        # IMFs to consider
# DC    = 0         # no DC part imposed
# init  = 1         # initialize omegas uniformly
# tol   = 1e-7
# """

# import_cy_module = f"import xarray as xr; from VMD import vmd; import numpy as np; {text}"
# import_py_module = f"import xarray as xr; from VMD_python import vmd as vmd_python; import numpy as np; {text}"

# cy_vmd = "vmd(f, alpha, tau, NIMF, DC, init, tol)"
# py_vmd = "vmd_python(f, alpha, tau, NIMF, DC, init, tol)"

# print("Cython", timeit.timeit(stmt=cy_vmd, setup=import_cy_module, number=100))
# print("Python", timeit.timeit(stmt=py_vmd, setup=import_py_module, number=100))
