import numpy as np

def cdfcalc(x0, disn, ind):
    """
    This function computes the Cumulative Distribution
    Function or CDF.

    Parameters
    ----------
    x0 : list or (1-D) ndarray of double
    disn : double
    ind : double

    Returns
    -------
    z : list or (1-D) ndarray of double
    """
    x     = np.asarray(x0, dtype=np.float64)
    lx    = len(x)
    ldisn = np.asarray(disn).flatten().size

    # Compute component's values of z
    z = np.empty(lx, dtype=np.float64)
    for i in range(lx):
        if (ind <= x[i]) == False:
            temp = 0.0
        else:
            if ldisn > 1:
                temp = max(disn)
            else:
                temp = disn

        z[i] = temp

    return z
