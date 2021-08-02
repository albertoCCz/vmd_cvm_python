# vmd_cvm_python

Algorithm for signal denoising according to the VMD_CVM method, which uses the Variational Mode Decomposition (VMD) to identify the relevant modes and the Cramer Von Mises statistics to filter out the noise.

### References
+ VMD_CVM:    Khuram Naveed, Muhammad Tahir Akhtar, Muhammad Faisal Siddiqui, Naveed ur Rehman, "A statistical approach to signal denoising based on data-driven multiscale representation", Digital Signal Processing, Vol. 108, pp. 102896, 2021.
+ VMD:        K. Dragomiretskiy and D. Zosso, "Variational Mode Decomposition," in IEEE Transactions on Signal Processing, vol. 62, no. 3, pp. 531-544, Feb.1, 2014, doi: 10.1109/TSP.2013.2288675.

### Usage
First we make the necessary imports:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vmd_cvm_python.Prop_VMD_CVM_python import Prop_VMD_CVM
```
In this example we are going to be using the Python version, so we import the package and the function with the "_python" ending. The `Prop_VMD_CVM` function executes all the algorithm at once, so that's all we need to import from this library in order to use this denoising method.

Then we fix the parameters for the function:
```python
# Input parameters
WIN_LEN = 256       # window length
NIMF    = 10        # number of modes
NP      = 500       # number of consecutive signal points

# Select Pfa using a decaying function e^(-k+1)
opt_Pfa = np.exp(-np.arange(0, NIMF-1))        # probability of false activation
```
It is very likely that you'll have to play around we these parameters when denoising your own signals. I recommend reading the References for an in deepth explanation of each of them.

Then we generete some signal, a sine in this example, an we add some noise to it:
```python
# Generate some signal
x      = np.linspace(-2*np.pi, 2*np.pi, 1000)
signal = np.sin(x)

# Add noise
seed  = 123456
rng   = np.random.default_rng(seed)
noise = rng.normal(loc=0.0, scale=1.0, size=x.shape[0])

noisy = signal + 1 * noise
```
Now we are ready to apply the VMD_CVM algorithm and try to denoise the signal. We do so like this:
```python
# Denoise using VMD_CVM method
_, _, denoised = Prop_VMD_CVM(noisy, WIN_LEN, NIMF, opt_Pfa, NP)
```
where we just passed the signal to denoise and the previously chosen parameters. The third returned element is actually the "denoised" signal. 

We can apply some simple post-processing like a rolling window that takes the mean to, sometimes, improve the result.

Let's plot all the three signals to see the result.
```python
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
```
![hello world denoising with VMD_CVM method](https://github.com/albertoCCz/vmd_cvm_python/blob/master/plots/hello_world.png)

Not bad!

The [full script](https://github.com/albertoCCz/vmd_cvm_python/blob/master/hello_world-vmd_cvm.py) is here:
```python
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
```