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

    # Create ECDF evaluation points array
    x = np.linspace(start=np.min(array), stop=np.max(array), num=len_array)

    y = np.empty(len_array, dtype=np.float64)
    for idx, elem in enumerate(x):
        y[idx] = sum(array <= elem) / len_array

    return y, x
