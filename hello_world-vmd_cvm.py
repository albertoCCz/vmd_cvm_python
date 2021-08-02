import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vmd_cvm_python.Prop_VMD_CVM_python import Prop_VMD_CVM

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

# Post-processing
denoised = pd.Series(denoised).rolling(window=100, min_periods=1, center=True).mean()

# Plot noisy, original and denoised signals
plt.subplots(1, figsize=(10,8))

plt.plot(x, noisy,    linewidth=0.8, label='noisy', alpha=0.5)
plt.plot(x, signal,   linewidth=0.8, label='original')
plt.plot(x, denoised, linewidth=0.8, label='denoised')

plt.title('Signal denoising by VMD_CVM method')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('plots/hello_world.png')
plt.show()
