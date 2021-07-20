import numpy as np
from VMD_python import vmd

def Prop_VMD_CVM(noisy, N, NIMF, Pf, Np):
    """
    This function implements a modified version of
    the VMD_CVM algorithm for signal denoising.

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
        Number of 
    

    """
    f = noisy

    pts = len(a)         # data length
    sigma = np.std(a-f)  # Estimate total noise variance for normalising IMFs

    # Some sample parameters for VMD
    alpha = 2000  # moderate badwidth constraint
    tau   = 0     # noise-tolerance (no strict fidelity enforcement)
    DC    = 0     # no DC part imposed
    init  = 1     # initialize omegas uniformly
    tol   = 1e-7

    # Variational mode decomposition of the noisy signal
    imf1, imf_hat, omega = vmd(f, alpha, tau, NIMF, DC, init, tol)

    return imf1, imf_hat, omega

