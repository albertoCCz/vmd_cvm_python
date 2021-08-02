import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cpdef double cvm(int z, int win_len):
    cdef int N = win_len
    cdef DTYPE_t[:] I = np.arange(1, N+1, 1, dtype=DTYPE)
    cdef int i
    cdef double result = 0.0

    for i in range(N):
        result += (z - (2.0 * I[i] - 1)/2.0/N)**2

    result += 1/12.0/N

    return result
