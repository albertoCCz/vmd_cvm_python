import xarray as xr
import numpy as np

data = xr.load_dataset('Data/2015_Granada_noisy.nc')

def myHist(x):
    return np.histogram(x, bins=10)

beta = np.array(data.beta_mean.groupby("time.hour").groups.values())
data_hist = np.apply_along_axis(myHist, axis=0, arr=)

print(f"data_hist:\n{data_hist}")
