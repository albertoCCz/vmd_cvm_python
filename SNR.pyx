import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef double snr(x0, y0):
    cdef DTYPE_t[:] x = np.asarray(x0, dtype=DTYPE)
    cdef DTYPE_t[:] y = np.asarray(y0, dtype=DTYPE)

    # Check both arrays have the same length
    assert len(x) == len(y)

    cdef double norm1 = 0.0
    cdef double norm2 = 0.0
    cdef int i
    cdef double result

    # Compute 2-norms
    for i in range(len(x)):
        norm1 += x[i]**2
        norm2 += (x[i] - y[i])**2

    norm1 = np.sqrt(norm1)
    norm2 = np.sqrt(norm2)

    # Compute result
    result = 20 * np.log10(norm1/norm2)

    return result
