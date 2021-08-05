import numpy as np
cimport numpy as np


cpdef cdfcalc(double[:] x, double[:] disn, ind):
    """
    This function computes the Cumulative Distribution
    Function or CDF for a particular threshold t:

    CDF(t) = 1/N * sum(y < t)

    where y is the sample for which we are computing the CDF, N is
    the length of y, and t is the threshold.

    Parameters
    ----------
    x : array_like 1D
        Sample for which we compute the CDF
    disn : double
        CDF values of x0
    ind : double

    Returns
    -------
    z : array_like 1D
    """
    cdef size_t lx = len(x)

    # Compute component's values of z
    cdef np.float64_t[:] z     = np.empty(shape=lx, dtype=np.float64)
    cdef np.float64_t    temp  = 0.0
    cdef size_t          i, j
    cdef np.int_t        start = 0
    for i in range(lx):
        for j in range(start, ind.shape[0]):
            if (ind[j] < x[i]):
                if temp < disn[j]:
                    temp = disn[j]
            else:
                start = j
                break

        z[i] = temp


    return z
