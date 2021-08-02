import numpy as np
cimport numpy as np

from VMD import vmd

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def Prop_VMD_CVM(ref, noisy, N, NIMF, Pf, Np):
    """
    """
    cdef DTYPE_t[:] a = np.asarray(ref, dtype=DTYPE)
    cdef DTYPE_t[:] f = np.asarray(noisy, dtype=DTYPE)

    cdef size_t pts  = len(a)

    cdef DTYPE_t sigma
    cdef DTYPE_t[:] temp = np.empty(pts, dtype=DTYPE)
    cdef size_t i
    for i in range(pts):
        temp[i] = a[i] - f[i]
    sigma = np.std(temp)
    del temp

    # Some sample parameters for VMD
    cdef int alpha   = 2000  # moderate badwidth constraint
    cdef int tau     = 0     # noise-tolerance (no strict fidelity enforcement)
    cdef int DC      = 0     # no DC part imposed
    cdef int init    = 1     # initialize omegas uniformly
    cdef DTYPE_t tol = 1e-7

    # Variational mode decomposition of the noisy signal
    imf1, imf_hat, omega = vmd(f, alpha, tau, NIMF, DC, init, tol)

    return imf1, imf_hat, omega
