import numpy as np

def mse(x0, y0):
    x = np.asarray(x0, dtype=np.float64)
    y = np.asarray(y0, dtype=np.float64)

    lx = len(x)
    ly = len(y)

    # Check both ndarrays have the same length
    assert lx == ly

    # Check ndarrays length is != 0
    assert lx != 0

    # Compute result
    result = 0
    for i in range(lx):
        result += (x[i] - y[i])**2
    
    result /= lx

    return result
