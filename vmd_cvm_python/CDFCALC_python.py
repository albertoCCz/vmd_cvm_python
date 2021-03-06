import numpy as np

def cdfcalc(x0, disn, ind):
    """
    This function computes the Cumulative Distribution
    Function or CDF for a particular threshold t:

    CDF(t) = 1/N * sum(y < t)

    where y is the sample for which we are computing the CDF, N is
    the length of y, and t is the threshold.

    Parameters
    ----------
    x0 : array_like 1D
        Sample for which we compute the CDF
    disn : array_like 1D
        CDF values of x0
    ind : array_like 1D

    Returns
    -------
    z : array_like 1D
    """
    x     = np.asarray(x0, dtype=np.float64)
    lx    = len(x)

    # Compute component's values of z
    z = np.empty(lx, dtype=np.float64)
    for i in range(lx):
        opts = disn[ind < x[i]]

        if opts.shape[0] < 1:
            temp = 0
        else:
            temp = np.max(opts)

        z[i] = temp

    return z
