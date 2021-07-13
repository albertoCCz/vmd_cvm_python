import numpy as np

def cvm(z, win_len):
    N = win_len
    I = np.arange(1, N+1, 1, dtype=np.int)

    result = 0.0
    for i in range(N):
        result += (z - (2 * I[i] - 1)/2/N)**2

    result += 1/12/N
    
    return result
