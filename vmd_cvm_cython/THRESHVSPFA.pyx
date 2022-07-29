import numpy as np
cimport numpy as np

from vmd_cvm_cython.cdfcalc import cdfcalc
from vmd_cvm_cython.cvm import cvm
from vmd_cvm_cython.ecdf import ecdf

def threshvspfa(np.float64_t[:] imfvec, int N):
    """
    This function computes the Thresholds values
    for each IMF.

    Parameters
    ----------
    imfvec : array_like 1D
        Flatten vector containing the noisy modes
    N : int
        Number of elements in each window

    Returns
    -------
    PfvThvec : array_like 2D
    ind_m : array_like 1D
    disn_m : array_like 1D
    """
    # Estimation of noise EDF from rejected modes
    # -------------------------------------------
    cdef int                MC      = len(imfvec)
    cdef np.int_t           windows = MC//N
    cdef np.float64_t[:]    ch      = np.empty(shape=N,            dtype=np.float64)
    cdef np.float64_t[:, :] tv      = np.empty(shape=(N, windows), dtype=np.float64)
    cdef np.float64_t[:, :] ti      = np.empty(shape=(N, windows), dtype=np.float64)
    cdef np.float64_t[:]    temp    = np.empty(shape=N,            dtype=np.float64)
    cdef np.float64_t[:]    tind    = np.empty(shape=N,            dtype=np.float64)
    cdef size_t i
    for i in range(windows):                  # loop for all windows
        ch[:]      = imfvec[N*i:N*(i+1)]      # pick the jth window
        temp, tind = ecdf(ch)                 # calculate ECDF
        tv[:, i]   = temp[:]                  # store value in tv
        ti[:, i]   = tind[:]                  # store index in ti

    cdef np.float64_t[:] disn_m = np.mean(tv, axis=1)
    cdef np.float64_t[:] ind_m  = np.mean(ti, axis=1)

    # Threshold versus Pfa curve estimation from rejected modes
    # ---------------------------------------------------------
    cdef np.float64_t    thresh_min = 0.001
    cdef np.float64_t    inc        = 0.001
    cdef np.float64_t    thresh_max = 20
    cdef np.float64_t[:] threshvec  = np.arange(thresh_min, thresh_max+inc, inc, dtype=np.float64)
    cdef np.float64_t[:] pfavec     = np.zeros(shape=threshvec.shape[0], dtype=np.float64)

    # Pre-compute cvm for each window
    cdef np.float64_t[:] tests = np.empty(shape=windows, dtype=np.float64)
    cdef np.float64_t[:] z     = np.empty(shape=N,       dtype=np.float64)
    cdef size_t litcount
    for litcount in range(windows):     # Loop through all windows
        z = cdfcalc(np.sort(imfvec[N*litcount:N*(litcount+1)]), disn_m, ind_m)
        tests[litcount] = cvm(z, N)     # Compute CVM distance between window ECDF and estimated ECDF for noise distribution
    
    # Compute probability of false detection of signal for threshold value
    cdef np.float64_t thresh, count_detection, Pfa
    cdef size_t       j
    for i, thresh in enumerate(threshvec):    # Loop through all candidate thresholds
        count_detection = 0
        for j in range(tests.shape[0]):       # Count the number of times this distance is not close-fit for a particular threshold
            if tests[j] > thresh:
                count_detection += 1

        Pfa = count_detection / windows       # Probability of false detection
        pfavec[i] = Pfa
        
        if Pfa < 0.000005:      # Lower bound of Pfa for sufficiently good threshold value
            break

    cdef np.float64_t[:, :] PfvThvec = np.asarray([threshvec, pfavec], dtype=np.float64)

    return PfvThvec, disn_m, ind_m
