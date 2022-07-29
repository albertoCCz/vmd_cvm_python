import numpy as np
cimport numpy as np

from vmd_cvm_cython.cdfcalc import cdfcalc
from vmd_cvm_cython.cvm import cvm
from vmd_cvm_cython.ecdf import ecdf
from vmd_cvm_cython.threshvspfa import threshvspfa
from vmdpy import VMD as vmd


cpdef Prop_VMD_CVM(noisy_py, N_py, NIMF_py, Pf_py, Np_py):
    """
    This function implements a slightly modified version
    of the VMD_CVM algorithm for signal denoising.

    See Also
    --------
    Original algorithm: 
    Khuram Naveed, Muhammad Tahir Akhtar, Muhammad Faisal Siddiqui, Naveed ur Rehman,
    "A statistical approach to signal denoising based on data-driven multiscale representation",
    Digital Signal Processing, Vol. 108, pp. 102896, 2021.

    Parameters
    ----------
    noisy_py : array_like 1D
        Noisy signal to denoise
    N_py : int
        Number of elements in each window
    NIMF_py : int
        The number of modes to be recovered
    Pf_py : array_like 1D
        Posible probabilities of false activation
    Np_py : int
        Number of consecutive iterations that must detect for the detection to hold
    
    Returns
    -------
    imf : array_like 2D
        Collection of decomposed modes
    rec : array_like 2D
        Recovered signal on IMF level
    sigrec : array_like 1D
        Recovered signal
    """
    # Check array-like params have correct dimesionaly
    assert len(noisy_py.shape) == 1 and \
           len(Pf_py.shape)    == 1

    # Params to C types
    cdef double[:] noisy = noisy_py
    cdef np.int_t  N     = N_py
    cdef size_t    NIMF  = NIMF_py
    cdef double[:] Pf    = Pf_py
    cdef np.int_t  Np    = Np_py
 
    cdef size_t pts = noisy.shape[0]  # data length

    # Some sample parameters for VMD
    cdef double alpha = 2000  # moderate bandwidth constraint
    cdef double tau   = 0     # noise-tolerance (no strict fidelity enforcement)
    cdef size_t  DC   = 0     # no DC part imposed
    cdef size_t init  = 1     # initialize omegas uniformly
    cdef double tol   = 1e-7

    # Variational mode decomposition of the noisy signal
    cdef double[:, :] imf
    imf = vmd(noisy_py, alpha, tau, NIMF, DC, init, tol)[0]

    if imf.shape[0] < NIMF:
        NIMF = imf.shape[0] - 1
        print('NIMF found to be less than the size of actual imf')
    
    cdef double[:, :] rec = np.zeros(shape=(imf.shape[0], imf.shape[1]), dtype=np.float64)

    # Determining k'_2 index
    # ----------------------

    # Compute distance between each mode and the estimated noisy ECD
    cdef double[:] Dist  = np.empty(shape=NIMF, dtype=np.float64)
    cdef double[:] tempx = np.empty(shape=imf.shape[1], dtype=np.float64)

    nEdf, nInd = ecdf(noisy)
    
    cdef double[:] z
    cdef size_t imfcnt
    for imfcnt in range(NIMF):
        tempx        = imf[imfcnt]
        z            = cdfcalc(np.sort(tempx), nEdf, nInd)
        Dist[imfcnt] = cvm(z, pts)

    cdef double[:] D = np.abs(np.diff(Dist))
    cdef np.int_t        n = np.argmax(D)
    while n <= NIMF/2.0:
        D[n] = 0.0
        if np.argmax(D) == n: break
        else: n = np.argmax(D)

    cdef np.int_t ni = NIMF - n
    if ni < 3:
        ni = 3

    # Compute thresholds
    # ------------------
    cdef double[:] imfvec = np.empty(shape=(imf.shape[0]-(NIMF-ni-1))*imf.shape[1], dtype=np.float64)
    cdef size_t i, j
    for i in range(NIMF-ni-1, imf.shape[0]):
        for j in range(imf.shape[1]):
            imfvec[(i-(NIMF-ni-1))*imf.shape[1]+j] = imf[i, j]

    PfvThvec, disn_m, ind_m = threshvspfa(imfvec, N)

    # Detection of signal/noisy coefficients in modes containing signal
    # -----------------------------------------------------------------
    cdef double[:] TH                        = PfvThvec[0, :]
    cdef double[:] PF                        = PfvThvec[1, :]
    cdef double[:] temp                      = np.empty(shape=imf.shape[1], dtype=np.float64)
    cdef np.int_t  indpf
    cdef double    thresh, elem, min_, temp_

    cdef double[:] x      = np.empty(shape=2*(N//2), dtype=np.float64)
    cdef double    test
    cdef size_t          ii, jj

    cdef np.int_t[:] bg             # Beginning of clusters of 1's (supossedly signal)
    cdef np.int_t[:] ed             # End of clusters

    for imfcnt in range(NIMF-ni+1):
        temp[:] = imf[imfcnt, :]                # Current IMF values
        indpf   = 0                             # Matches the optimal Pfa calulated for current IMF with available PFAs
        for ii, elem in enumerate(PF):
            temp_ = np.abs(Pf[imfcnt] - elem)
            if ii == 0:
                min_ = temp_
            elif temp_ < min_:
                min_ = temp_
                indpf = ii

        thresh  = TH[indpf]                     # Extract the threshold value for current IMF
        
        # Signal/noise discrimination in each window
        booln  = np.zeros(shape=temp.shape[0], dtype=int)
        for jj in range(N, pts-N):
            x    = temp[(jj - N//2):(jj + N//2)]
            z    = cdfcalc(np.sort(x), disn_m, ind_m)

            test = cvm(z, N)        # CVM statistic
            if test > thresh:       # statistic > threshold: signal present # ***WARNING***
                booln[jj] = 1

        # Consider detection only if it happens at least for length N; removes impulse-like noise!
        D_ = np.diff(np.pad(booln, (1,1), 'constant', constant_values=(0,0)))      # Find "edges"
        bg = np.nonzero(D_ == 1)[0].astype(int)         # Beginning of clustes of 1's (supossedly signal)
        ed = np.nonzero(D_ == -1)[0].astype(int) - 1    # End of clusters
        for ii in range(len(bg)):
            if ((ed[ii] - bg[ii]) < Np):                   # If the length of cluster is too small we attribute detection to noise peak
                booln[bg[ii]:ed[ii]] =  0                  # No detection of signal there

        # Effectively eliminate detected noise peaks
        for i in range(booln.shape[0]):
            rec[imfcnt, i] = temp[i] * booln[i]
        
    rec[0] = imf[0]
    cdef double[:] sigrec = np.sum(rec, axis=0)


    return imf, rec, sigrec
