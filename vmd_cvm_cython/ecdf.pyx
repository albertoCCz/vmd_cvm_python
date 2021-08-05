import numpy as np
cimport numpy as np


cpdef tuple ecdf(np.float_t[:] array):
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
    cdef size_t len_array = len(array)

    # Create ECDF evaluation points array
    cdef np.float64_t[:] x = np.linspace(start=np.min(array), stop=np.max(array), num=len_array)

    # Compute ECDF values at x
    cdef np.float64_t[:] y = np.empty(len_array, dtype=np.float64)
    cdef np.float64_t    count
    cdef np.float64_t    point
    cdef size_t          idx
    cdef np.float64_t    elem
    for idx, elem in enumerate(x):
        count = 0
        for point in array:
            if point <= elem:
                count += 1
        y[idx] = count / len_array
    
    return y, x
