import numpy as np

def cvm(z, win_len):
    """
    Computes the Cramer Von Mises
    statistical distance.

    Parameters
    ----------
    z : array_like
    win_len : int

    Returns
    -------
    result : float
        Cramer Von Mises statistical
        distance
    """
    N = win_len
    I = np.arange(1, N+1, 1, dtype=np.int)
    
    result = np.sum((z - (2 * I - 1)/2/N)**2) + 1/12/N
        
    return result
