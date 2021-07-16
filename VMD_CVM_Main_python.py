# import matplotlib.pyplot as plt
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
print(f"Are imfs the same: {imf == imf_p}")
print(f"Are recs the same: {rec == rec_p}")
print(f"Are ys the same:   {y == y_p}")

# Plot reconstructed signal

# print(end-start)
# heights = xr.load_dataset('Data/2015_Granada_Noisy.nc')['range'].values
# plt.plot(np.sum(imf, axis=0), heights, linewidth=0.6, label="Reconstructed")
# plt.grid()
# plt.xlabel("RCS [arb. units]")
# plt.ylabel("Range [m]")
# plt.title("Reconstructed signal using VMD method")
# plt.show()
