import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef DTYPE_t[:] cdfcalc(x0, double disn, double ind):
    """
    This function computes the Cumulative Distribution
    Function or CDF.

    Parameters
    ----------
    x0 : list or (1-D) ndarray of double
    disn : double
    ind : double

    Returns
    -------
    z : DTYPE_t[:]
    """
    cdef DTYPE_t[:] x = np.asarray(x0, dtype=DTYPE)
    cdef size_t lx = len(x)
    cdef size_t ldisn = np.asarray(disn).flatten().size
    
    cdef DTYPE_t[:] z = np.empty(lx, dtype=DTYPE)
    cdef size_t i
    cdef DTYPE_t temp

    # Compute component's values of z
    for i in range(lx):
        if (ind <= x[i]) == False:
            temp = 0.0
        else:
            if ldisn > 1:
                temp = max(disn)
            else:
                temp = disn
        
        z[i] = temp
    
    return z
