import numpy as np

from VMD_python import vmd
from ecdf_python import ecdf
from CDFCALC_python import cdfcalc
from CVM_python import cvm
from THRESHVSPFA_python import threshvspfa

import warnings
warnings.filterwarnings("always", UserWarning)


def Prop_VMD_CVM(noisy, N, NIMF, Pf, Np):
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
    noisy : array_like 1D
        Noisy signal to denoise
    N : int
        Number of elements in each window
    NIMF : int
        The number of modes to be recovered
    Pf : array_like 1D
        Posible probabilities of false activation
    Np : int
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
    pts = len(noisy)  # data length

    # Some sample parameters for VMD
    alpha = 2000  # moderate badwidth constraint
    tau   = 0     # noise-tolerance (no strict fidelity enforcement)
    DC    = 0     # no DC part imposed
    init  = 1     # initialize omegas uniformly
    tol   = 1e-7

    # Variational mode decomposition of the noisy signal
    imf, _, _ = vmd(noisy, alpha, tau, NIMF, DC, init, tol)

    if imf.shape[0] < NIMF:
        NIMF = imf.shape[0] - 1
        warnings.warn("NIMF found to be less than the size of actual imf")

    rec = np.zeros(shape=imf.shape, dtype=np.float64)

    # Determining k'_2 index
    # ----------------------
    
    # Compute distance between each mode and the estimated noisy ECD
    Dist  = np.empty(shape=NIMF, dtype=np.float64)
    tempx = np.empty(shape=imf.shape[1], dtype=np.float64)
    nEdf, nInd = ecdf(noisy)
    for imfcnt in range(NIMF):
        tempx        = imf[imfcnt, :]
        # Edf, Ind     = ecdf(tempx)
        z            = cdfcalc(np.sort(tempx), nEdf, nInd)
        Dist[imfcnt] = cvm(z, pts)

    D = np.abs(np.diff(Dist))
    n = np.argmax(D)
    while n <= 5:
        D[n] = 0
        n    = np.argmax(D)
    
    ni = 10 - n
    if ni < 3:
        ni = 3
    
    # Compute thresholds
    # ------------------
    imfvec = np.empty(shape=imf.shape[0]*imf.shape[1], dtype=np.float64)
    imfvec = imf.flatten()

    PfvThvec, disn_m, ind_m = threshvspfa(imfvec, N)


    # Detection of signal/noisy coefficients in modes containing signal
    # -----------------------------------------------------------------
    booln = np.zeros(shape=imf.shape[1], dtype=np.int)
    for imfcnt in range(NIMF-3):
        TH = np.transpose(PfvThvec[0, :])
        PF = np.transpose(PfvThvec[1, :])
        temp = imf[imfcnt, :]                         # Current IMF values
        indpf = np.argmin(np.abs(Pf[imfcnt] - PF))    # Matches the optimal Pfa calulated for current IMF with available PFAs

        thresh = TH[indpf]

        # Signal/noise discrimination in each window
        for jj in range(N//2, pts-N//2):
            x = temp[(jj - N//2):(jj + N//2+1)]
            z = cdfcalc(np.sort(x), disn_m, ind_m)

            test = cvm(z, N)    # CVM statistic
            if test > thresh:   # statistic > threshold: signal present
                booln[jj] = 1

        # Consider detection only if it happens at least for length N; removes impulse-like noise! 
        D = np.diff(np.asarray([0, booln, 0]))   # Find "edges"
        bg = np.nonzero(D == 1)                  # Beginning of clustes of 1's (supossedly signal)
        ed = np.nonzero(D == -1)                 # End of clusters
        for ii in range(len(bg)):
            if (ed[ii] - bg[ii] < Np):           # If the length of cluster is too small we attribute detection to noise peak
                booln[bg[ii]:ed[ii]] = 0         # No detection of signal there

        # Effectively eliminite detected noise peaks
        rec[imfcnt, :] = temp * booln

        rec[0] = imf[0]
        sigrec = np.sum(rec, axis=0).transpose()
    

    return imf, rec, sigrec
