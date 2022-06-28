import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    from vmd_cvm_python.Prop_VMD_CVM_python import Prop_VMD_CVM
elif len(sys.argv) == 2:
    if sys.argv[1] == '-python':
        from vmd_cvm_python.Prop_VMD_CVM_python import Prop_VMD_CVM
    elif sys.argv[1] == '-cython':
        from vmd_cvm_cython.Prop_VMD_CVM import Prop_VMD_CVM
    
    # Implementation modo
    MODE = sys.argv[1].lstrip('-')
else:
    print("[ERROR]: Wrong number of arguments passed.\n" + \
          "Usage:\n" + \
          "    python hello_world-vmd_cvm.py [OPTIONS]\n" + \
          "where\n" + \
          "    OPTIONS          Explanation\n" + \
          "    -python          To run the example using the Python implementation\n" + \
          "    -cython          To run the example using the Cython implementation\n", sep='')
    exit(1)

# Input parameters
WIN_LEN = 256       # window length
NIMF    = 10        # number of modes
NP      = 500       # number of consecutive signal points

# Select Pfa using a decaying function e^(-k+1)
opt_Pfa = np.exp(-np.arange(0, NIMF-1))          # probability of false activation

# Generate some signal
x      = np.linspace(-2*np.pi, 2*np.pi, 1000)
signal = np.sin(x)

# Add noise
seed  = 123456
rng   = np.random.default_rng(seed)
noise = rng.normal(loc=0.0, scale=1.0, size=x.shape[0])

noisy = signal + 1 * noise

# Denoise using VMD_CVM method
_, _, denoised = Prop_VMD_CVM(noisy, WIN_LEN, NIMF, opt_Pfa, NP)

if MODE == 'cython':
    denoised = np.array(denoised.copy(), dtype=np.float64)

# Post-processing
denoised = pd.Series(denoised).rolling(window=100, min_periods=1, center=True).mean()

# Plot noisy, original and denoised signals
plt.subplots(1, figsize=(10,8))

plt.plot(x, noisy,    linewidth=0.8, label='noisy', alpha=0.5)
plt.plot(x, signal,   linewidth=0.8, label='original')
plt.plot(x, denoised, linewidth=0.8, label='denoised')

plt.title(f'Signal denoising by VMD_CVM method ({MODE.capitalize()} impl.)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('plots/hello_world.png')
plt.show()
