import numpy as np
cimport numpy as np

DTYPE = np.float64
DTYPE_complex = np.complex

ctypedef np.float64_t DTYPE_t
ctypedef np.complex_t DTYPE_complex_t

def vmd(signal, alpha, tau, size_t K, DC, init, tol):
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
    cdef size_t len_freqs = len(freqs)
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
    f_hat_plus[:T//2]                  = np.zeros(T//2, dtype=DTYPE_complex)

    # Matrix keeping track of every iterant // could be discarded for mem
    cdef DTYPE_complex_t[:, :, :] u_hat_plus =  np.zeros((N, len_freqs, K), dtype=np.complex)

    # Initialization of omega_k
    cdef DTYPE_t[:, :] omega_plus = np.zeros((N, K), dtype=np.float64)
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5/K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.random(K))).astype(DTYPE)
    else:
        omega_plus[0, 0] = 0.0

    # Start with empty dual variables
    cdef DTYPE_complex_t[:, :] lambda_hat = np.zeros((N, len_freqs), dtype=DTYPE_complex)

    # Other inits
    cdef DTYPE_t            eps    = np.spacing(DTYPE(1.0))                    # distance from 1.0 to the next largest double-precision number.
    cdef DTYPE_t            uDiff  = tol + eps                                 # update step
    cdef size_t             n      = 0                                         # loop counter
    cdef size_t             k      = 0
    cdef DTYPE_complex_t[:] sum_uk = np.zeros(len_freqs, dtype=DTYPE_complex)  # accumulator
    cdef DTYPE_t[:]         temp   = np.empty(len_freqs, dtype=DTYPE)
    cdef DTYPE_complex_t[:] temp2  = np.empty(len_freqs, dtype=DTYPE_complex)

    # Main loop for iterative updates
    # -------------------------------

    while (uDiff > tol and n < N):
        # Update first mode accumulator
        sum_uk = (np.real(sum_uk) + np.real(u_hat_plus[n, :, K-1]) - np.real(u_hat_plus[n, :, 0])) + \
                 (np.imag(sum_uk) + np.imag(u_hat_plus[n, :, K-1]) - np.imag(u_hat_plus[n, :, 0])) * 1j
        
        # Update spectrum of first mode through Wiener filter of residuals
        for i in range(len_freqs):
            temp[i] = 1 + Alpha[k] * (freqs[i] - omega_plus[n, k])**2
        
        u_hat_plus[n+1, i, k] = ((np.real(f_hat_plus) - np.real(sum_uk) - np.real(lambda_hat[n, :])/2) + \
                                 (np.imag(f_hat_plus) - np.imag(sum_uk) - np.imag(lambda_hat[n, :])/2) * 1j) \
                                / temp

        # Update first omega if not held at 0
        if not DC:
            omega_plus[n+1, k] = ((freqs[T//2:T]) * np.matrix(np.abs(u_hat_plus[n+1, T//2:T, k])**2).H) \
                                 / np.sum(np.abs(u_hat_plus[n+1, T//2:T, k])**2)

        # Update of any other mode
        for k in range(1, K):
            # Accumulator
            sum_uk += (np.real(u_hat_plus[n+1, :, k-1]) - np.real(u_hat_plus[n, :, k])) + \
                      (np.imag(u_hat_plus[n+1, :, k-1]) - np.imag(u_hat_plus[n, :, k])) * 1j
        
            # mode spectrum
            for i in range(len_freqs):
                temp[i] = 1 + Alpha[k] * (freqs[i] - omega_plus[n, k])**2

            u_hat_plus[n+1, :, k] = ((np.real(f_hat_plus) - np.real(sum_uk) - np.real(lambda_hat[n, :])/2) + \
                                     (np.imag(f_hat_plus) - np.imag(sum_uk) - np.imag(lambda_hat[n, :])/2) * 1j) \
                                    / temp

            # Center frequencies
            omega_plus[n+1, k] = (freqs[T//2:T] * np.matrix((np.abs(u_hat_plus[n+1, T//2:T, k]))**2).H) \
                                 / np.sum(np.abs(u_hat_plus[n+1, T//2:T, k])**2)
        
        # Dual ascent
        lambda_hat[n+1, :] = lambda_hat[n, :] + tau * (u_hat_plus[n+1, :, :].sum(axis=1) - f_hat_plus)

        # Loop counter
        n += 1

        # Converged yet?
        uDiff = eps
        for i in range(K):
            temp2 = (np.real(u_hat_plus[n, :, i]) - np.real(u_hat_plus[n-1, :, i])) + \
                    (np.imag(u_hat_plus[n, :, i]) - np.imag(u_hat_plus[n-1, :, i])) * 1j
            uDiff += 1/T * np.sum(temp2 * temp2.transpose())

        uDiff = np.abs(uDiff)

    # Postprocessing and cleanup
    # --------------------------

    # Discard empty space if converged early
    N = min(N, n+1)
    cdef DTYPE_t[:, :] omega = omega_plus[:N, :]

    # Signal reconstruction
    cdef DTYPE_complex_t[:, :] u_hat = np.zeros((T, K), dtype=DTYPE_complex)
    u_hat[T//2:T, :]   = np.squeeze(u_hat_plus[N-1, T//2:T, :])
    u_hat[1:T//2+1, :] = np.flip(np.squeeze(u_hat_plus[N-1, T//2:T, :].conjugate()))
    u_hat[0, :]        = u_hat[-1, :].conjugate()

    cdef DTYPE_t[:, :] u = np.zeros((K, len(t)))

    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # Remove mirror part
    u = u[:, T//4:3*T//4]

    # Recompute spectrum
    del u_hat
    cdef DTYPE_complex_t[:, :] u_hat_ = np.empty(shape=(u.shape[1], K), dtype=DTYPE_complex)
    for k in range(K):
        u_hat_[:, k] = np.matrix(np.fft.fftshift(np.fft.fft(u[k, :]))).H.flatten()


    return u, u_hat_, omega
