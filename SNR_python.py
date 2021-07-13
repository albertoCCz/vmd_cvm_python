import numpy as np

def snr(x0, y0):
    x = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64)

    # Check both ndarrays have the same length
    assert len(x) == len(y)

    # Compute 2-norms
    norm1 = 0
    norm2 = 0
    for i in range(len(x)):
        norm1 += x[i]**2
        norm2 += (x[i] - y[i])**2

    norm1 = np.sqrt(norm1)
    norm2 = np.sqrt(norm2)
    
    # Compute result
    result = 20 * np.log10(norm1/norm2)

    return result
