import numpy as np

from ecdf_python import ecdf
from CDFCALC_python import cdfcalc
from CVM_python import cvm

def threshvspfa(imfvec, N):
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
    MC      = len(imfvec)
    windows = MC//N     # Number of windows
    ch      = np.empty(shape=N, dtype=np.float64)
    tv      = np.empty(shape=(N, windows), dtype=np.float64)
    ti      = np.empty(shape=(N, windows), dtype=np.float64)
    for i in range(windows):                  # loop for all windows
        ch         = imfvec[N*i:N*(i+1)]      # pick the jth window
        temp, tind = ecdf(ch)                 # calculate ECDF
        tv[:, i]   = temp                     # store value in tv
        ti[:, i]   = tind                     # store index in ti

    disn_m = np.mean(tv, axis=1)
    ind_m  = np.mean(ti, axis=1)
     
    # Threshold versus Pfa curve estimation from rejected modes
    # ---------------------------------------------------------
    thresh_min = 0.001
    inc        = 0.001
    thresh_max = 20
    threshvec  = np.arange(thresh_min, thresh_max, inc, dtype=np.float64)
    pfavec     = np.zeros(shape=len(threshvec), dtype=np.float64)

    i = 0
    for thresh in threshvec:    # Loop through all candidate thresholds
        count_detection = 0

        for litcount in range(windows):   # Loop through all windows
            z = cdfcalc(np.sort(imfvec[N*litcount:N*(litcount+1)]), disn_m, ind_m)
            
            test = cvm(z, N)     # Compute CVM distance between window ECDF and estimated ECDF for noise distribution
            if test < thresh:    # Count the number of times this distance is not close-fit for a particular threshold
                count_detection += 1
            
        Pfa = count_detection / (windows)    # Compute probability of false detection of signal
        pfavec[i] = Pfa
        
        if Pfa < 0.000005:       # Lower bound of Pfa for sufficiently good threshold value
            break
    
        i += 1
    
    PfvThvec = np.asarray([threshvec, pfavec], dtype=np.float64)

    return PfvThvec, disn_m, ind_m
