import numpy as np

def threshvspfa(imfvec, N):
    """
    This function computes the Thresholds values
    for each IMF.

    Parameters
    ----------
    imfvec : list or (1-D) ndarray of double
    N : int

    Returns
    -------
    PfvThvec : 2-D ndarray of double
    ind_m : double
    disn_m : double
    """
    MC = len(imfvec)
    for j in range(MC // N):
        ch = imfvec[(N * (j-1) + 1):N*j]

