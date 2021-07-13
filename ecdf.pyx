import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef ecdf(data):
    """
    This function computes the Empirical
    Cumulative Distribution Function (ECDF)
    from a sample.

    Parameters
    ----------
    data : array-like

    Returns
    -------
    y : ndarray
        ECDF values evaluated at x
    x : ndarray
        ECDF evaluation points
    """
    cdef DTYPE_t[:] array = np.asarray(data, dtype=DTYPE) 
    cdef size_t len_array = len(array)

    # Unique values in sample array
    cdef DTYPE_t[:] cdfx = np.unique(array).astype(DTYPE)
    cdef size_t len_cdfx = len(cdfx)

    # Create ECDF evaluation points array
    cdef DTYPE_t[:] x = np.linspace(start=np.min(cdfx), stop=np.max(cdfx), num=len_cdfx)

    # Compute ECDF values at x
    cdef DTYPE_t[:] y = np.empty(len_cdfx, dtype=DTYPE)
    cdef DTYPE_t count
    cdef DTYPE_t point
    for idx, elem in enumerate(x):
        count = 0
        for point in array:
            if point <= elem:
                count += 1
        y[idx] = count / len_array
    
    return y, x
