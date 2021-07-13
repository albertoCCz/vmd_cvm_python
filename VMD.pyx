import numpy as np
cimport numpy as np

DTYPE = np.float64
DTYPE_complex = np.complex

ctypedef np.float64_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t

def vmd(signal, alpha, tau, K, DC, init, tol):
    """This function implements the Variational
    Mode Decomposition algorithm.

    See Also
    --------
    K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
    on Signal Processing (in press)

    Parameters
    ----------
    signal : array_like 1D
        the time domain signal (1D) to be decomposed
    alpha : float
        the balancing parameter of the data-fidelity constraint
    tau : float
        time-step of the dual ascent ( pick 0 for noise-slack )
    K : int
        the number of modes to be recovered
    DC : boolean or {0, 1}
        true if the first mode is put and kept at DC (0-freq)
    init : {0, 1, 2}
        0 = all omegas start at 0
        1 = all omegas start uniformly distributed
        2 = all omegas initialized randomly
    tol : float
        tolerance of convergence criterion; typically around 1e-6
    
    Returns
    -------
    u : array_like 2D
        the collection of decomposed modes
    u_hat : array_like 2D
        spectra of the modes
    omega : array_like 2D
        estimated mode center-frequencies
    """
    # Preparations
    # ------------

    # Period and sampling frequency of input signal
    cdef size_t save_T = len(signal)
    cdef DTYPE_t fs = 1 / save_T

    # Extend the signal by mirroring
    cdef size_t T = save_T
    cdef DTYPE_t[:] f_mirror = np.empty(shape=2*T, dtype=DTYPE)
    f_mirror[:T//2]       = np.flip(signal[:T//2])
    f_mirror[T//2:3*T//2] = signal
    f_mirror[3*T//2:]     = np.flip(signal[T//2:])
    cdef DTYPE_t[:] f     = f_mirror

    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    cdef DTYPE_t[:] t = (np.arange(1, T+1, 1) / T).astype(DTYPE)

    # Spectral Domain discretization
    cdef DTYPE_t[:] freqs = np.empty(len(t), dtype=DTYPE)
    cdef size_t i
    for i in range(len(t)):
        freqs[i] = t[i] - 0.5 - 1/T
    
    # Maximun number of iterations (if not converged yet, then it won't anyway)
    cdef int N = 500

    # For future generalizations: individual alpha for each mode
    cdef DTYPE_t[:] Alpha = np.ones(K, dtype=DTYPE)
    for i in range(K):
        Alpha[i] *= alpha

    # Construct and center f_hat
    cdef DTYPE_complex_t[:] f_hat      = np.fft.fftshift(np.fft.fft(np.asarray(f, dtype=DTYPE))).astype(DTYPE_complex)
    cdef DTYPE_complex_t[:] f_hat_plus = f_hat
    f_hat_plus[:T//2]                  = 0

    # Matrix keeping track of every iterant // could be discarded for mem
    cdef DTYPE_complex_t[:, :, :] u_hat_plus =  np.zeros((N, len(freqs), K), dtype=np.complex)

    # Initialization of omega_k
    cdef DTYPE_t[:, :] omega_plus = np.zeros((N, K), dtype=np.float64)
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5/K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.random(K))).astype(DTYPE)
        