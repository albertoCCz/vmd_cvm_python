import numpy as np

def ecdf(array):
    """
    This function computes the Empirical
    Cumulative Distribution Function (ECDF)
    from a sample.

    Parameters
    ----------
    array : array-like

    Returns
    -------
    y : array_like 1D
        ECDF values evaluated at x
    x : array_like 1D
        ECDF evaluation points
    """
    len_array = len(array)
    
    # Unique values in sample array
    cdfx      = np.unique(array).astype(np.float64)
    len_cdfx  = len(cdfx)
    
    # Create ECDF evaluation points array
    x = np.linspace(start=np.min(cdfx), stop=np.max(cdfx), num=len_cdfx)

    y = np.empty(len_cdfx, dtype=np.float64)
    for idx, elem in enumerate(x):
        y[idx] = sum(array <= elem) / len_array

    return y, x
