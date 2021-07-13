import numpy as np
from VMD_python import vmd

def Prop_VMD_CVM(ref, noisy, N, NIMF, Pf, Np):
    """
    """
    a = ref
    f = noisy

    pts = len(a)
    sigma = np.std(a-f)

    # Some sample parameters for VMD
    alpha = 2000  # moderate badwidth constraint
    tau   = 0     # noise-tolerance (no strict fidelity enforcement)
    DC    = 0     # no DC part imposed
    init  = 1     # initialize omegas uniformly
    tol   = 1e-7

    # Variational mode decomposition of the noisy signal
    imf1, imf_hat, omega = vmd(f, alpha, tau, NIMF, DC, init, tol)

    return imf1, imf_hat, omega

