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
from ecdf import ecdf
from ecdf_python import ecdf as ecdf_python

# Check results
array = [101, 118, 121, 103, 142, 111, 119, 122, 128, 112, 117,157]

y, x     = ecdf(array)
y_p, x_p = ecdf_python(array)

print("Cython\n", f"\nx: {list(x)}\ny: {list(y)}")
print("\nPython\n", f"\nx: {x_p}\ny: {y_p}")
"""

"""VMD
import numpy as np
import xarray as xr

from Prop_VMD_CVM import Prop_VMD_CVM
from Prop_VMD_CVM_python import Prop_VMD_CVM as Prop_VMD_CVM_python

# Input parameter
win_len = 32 # Window length
NIMF    = 10 # IMFs to consider
Np      = 36 # No of consecutive iterations that must detect for the detection to hold
N_mon   = 10
sigL    = 12

# Select Pfa using a decaying function e^(-k+1)
opt_Pfa = np.exp(-np.arange(0, NIMF, 1))

# Load signal
a  = xr.load_dataset('Data/2015_Granada_Noisy.nc').beta_mean.values[0]
a1 = xr.load_dataset('Data/2015_Granada_Denoised.nc').beta_mean.values[0]

# Generating the noisy signal for given input SNR
iSNR = 20
SNRii = iSNR

sigma = 0.1 + (0.3 / (SNRii + 1))
x     = np.arange(-1/2, 1/2, 1/(win_len-1))
g     = np.exp(- x**2 / (2 * sigma**2))
g     /= sum(g)

snri = iSNR

f = a

# Proposed approach
imf, rec, y = Prop_VMD_CVM(a, f, win_len, NIMF, opt_Pfa, Np)
imf_p, rec_p, y_p = Prop_VMD_CVM_python(a, f, win_len, NIMF, opt_Pfa, Np)

# Check results are the same
print(f"Are imfs the same: {sum(sum(imf == imf_p))/(imf_p.shape[0] * imf_p.shape[1]) == True}")
print(f"Are recs the same: {sum(sum(rec == rec_p))/(rec_p.shape[0] * rec_p.shape[1]) == True}")
print(f"Are ys the same:   {sum(sum(y == y_p))/(y_p.shape[0] * y_p.shape[1]) == True}")
"""
import numpy as np
import xarray as xr

from Prop_VMD_CVM import Prop_VMD_CVM
from Prop_VMD_CVM_python import Prop_VMD_CVM as Prop_VMD_CVM_python

import matplotlib.pyplot as plt

# Input parameter
win_len = 32 # Window length
NIMF    = 10 # IMFs to consider
Np      = 36 # No of consecutive iterations that must detect for the detection to hold

# Select Pfa using a decaying function e^(-k+1)
opt_Pfa = np.exp(-np.arange(0, NIMF, 1))

# Load signal
a  = xr.load_dataset('Data/2015_Granada_Noisy.nc').beta_mean.values[0].astype(np.float64)
range_ = xr.load_dataset('Data/2015_Granada_Noisy.nc')['range']

f = a

# Proposed approach
imf,   imf_hat,   omega   = Prop_VMD_CVM(a, f, win_len, NIMF, opt_Pfa, Np)
imf_p, imf_hat_p, omega_p = Prop_VMD_CVM_python(a, f, win_len, NIMF, opt_Pfa, Np)

# Plot results
# fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

# ax[0].plot(sum(imf), range_, color='cornflowerblue', linewidth=0.7, label='Cython')
# ax[0].set_xlabel('signal [arb. units]')
# ax[0].set_ylabel('range [m]')
# ax[0].set_title('Cython reconstruction')
# ax[0].grid()

# ax[1].plot(sum(imf_p), range_, color='forestgreen', linewidth=0.7, label='Python')
# ax[1].set_xlabel('signal [arb. units]')
# ax[1].set_title('Python reconstruction')
# ax[1].grid()

# plt.show()

# Check results are the same
print(f"Are imfs the same: {imf.shape == imf_p.shape}")
print(f"Are recs the same: {imf_hat.shape == imf_hat_p.shape}")
print(f"Are ys the same:   {omega.shape == omega_p.shape}")
