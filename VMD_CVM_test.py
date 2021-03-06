import numpy as np
from time import time
import xarray as xr
import matplotlib.pyplot as plt

from vmd_cvm_cython.Prop_VMD_CVM import Prop_VMD_CVM

# Input parameters
win_len = 32
NIMF    = 10
Np      = 36

# Select Pfa using a decaying function e^(-k+1)
opt_Pfa = np.exp(-np.arange(0, NIMF-1))

# Load data
print('\n\nInput index of signal to denoise {0, 1, 2, ...}: ')
idx = int(input())
noisy    = xr.load_dataset('Data/2015_Granada_noisy.nc').beta_mean.values[idx].astype(np.float64)
notnoisy = xr.load_dataset('Data/2015_Granada_denoised.nc').beta_mean.values[idx]

# Execute proposed algorithm
start = time()
_, _, denoised = Prop_VMD_CVM(noisy, win_len, NIMF, opt_Pfa, Np)

# Print execution time
print(f"\n\nExecution time:\t{int((time() - start)*10e3)/10e3} (s)")

# Plot noisy, not-noisy and denoised signals
range_ = xr.load_dataset('Data/2015_Granada_noisy.nc')['range'].values

fig, ax = plt.subplots(1, figsize=(10,8))

plt.plot(noisy,    range_, linewidth=0.8, label='noisy', alpha=0.5)
plt.plot(notnoisy, range_, linewidth=0.8, label='original')
plt.plot(denoised, range_, linewidth=0.8, label='denoised')

plt.xlabel('intensity [arb. units]')
plt.ylabel('range [m]')
plt.title('Ceilometer signal denoising')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
