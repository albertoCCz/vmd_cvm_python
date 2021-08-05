import numpy as np
cimport numpy as np


cpdef cvm(np.float64_t[:] z, size_t win_len):
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
    cdef int             N      = win_len
    cdef np.float64_t[:] I      = np.arange(1, N+1, 1, dtype=np.float64)
    cdef np.float64_t    result = 0.0
    cdef size_t          i
    for i in range(N):
        result += (z[i] - (2.0 * I[i] - 1)/2.0/N)**2

    result += 1/12.0/N

    return result
