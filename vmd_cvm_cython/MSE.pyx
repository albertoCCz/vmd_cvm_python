import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cpdef double mse(x0, y0):
    cdef DTYPE_t[:] x = np.asarray(x0, dtype=DTYPE)
    cdef DTYPE_t[:] y = np.asarray(y0, dtype=DTYPE)

    cdef size_t lx = len(x)
    cdef size_t ly = len(y)

    cdef double result = 0.0
    cdef size_t i

    # Check both ndarrays have the same length
    assert lx == ly

    # Check ndarrays length is != 0
    assert lx != 0

    for i in range(lx):
        result += (x[i] - y[i])**2

    result /= lx

    return result
