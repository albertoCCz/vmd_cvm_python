"""CVM
from CVM import cvm
from CVM_python import cvm as cvm_python

# Check results
print("Input z:")
z = int(input())
print("Input win_len:")
win_len = int(input())

print("Cython", cvm(z, win_len))
print("Python", cvm_python(z, win_len))
"""

"""SNR
from SNR import snr
from SNR_python import snr as snr_python

# Check results
x = [0,1,2,3]
y = [4,5,6,7]

print("Cython", snr(x, y))
print("Python", snr_python(x, y))
"""

"""MSE
from MSE import mse
from MSE_python import mse as mse_python

# Check results
x = [0,1,2,3,8,6,4,1,2,9,7,3,5,5,6,2,1,9,0,8]
y = [4,5,6,7,1,8,7,2,9,8,4,2,6,8,8,4,1,6,4,7]

print("Cython", mse(x, y))
print("Python", mse_python(x, y))
"""

"""CDFCALC
from CDFCALC import cdfcalc
from CDFCALC_python import cdfcalc as cdfcalc_python

# Check results
x    = [0,1,2,3,8,6,4,1,2,9,7]
disn = 7
ind  = 4

print("Cython", list(cdfcalc(x, disn, ind)))
print("Python", cdfcalc_python(x, disn, ind))
"""

"""ecdf
"""
from ecdf import ecdf
from ecdf_python import ecdf as ecdf_python

# Check results
array = [101, 118, 121, 103, 142, 111, 119, 122, 128, 112, 117,157]

y, x     = ecdf(array)
y_p, x_p = ecdf_python(array)

print("Cython\n", f"\nx: {list(x)}\ny: {list(y)}")
print("\nPython\n", f"\nx: {x_p}\ny: {y_p}")
