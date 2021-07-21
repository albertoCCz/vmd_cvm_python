import numpy as np

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
        Time domain signal (1D) to be decomposed
    alpha : float
        Balancing parameter of the data-fidelity constraint
    tau : float
        Time-step of the dual ascent ( pick 0 for noise-slack )
    K : int
        Number of modes to be recovered
    DC : boolean or {0, 1}
        True if the first mode is put and kept at DC (0-freq)
    init : {0, 1, 2}
        0 = all omegas start at 0
        1 = all omegas start uniformly distributed
        2 = all omegas initialized randomly
    tol : float
        Tolerance of convergence criterion; typically around 1e-6
    
    Returns
    -------
    u : array_like 2D
        Collection modes in which the signal has been decomposed
    u_hat : array_like 2D
        Spectra of the modes
    omega : array_like 2D
        Estimated mode center-frequencies
    """
    # Preparations
    # ------------

    # Period and sampling frequency of input signal
    save_T = len(signal)
    fs = 1 / save_T
    
    # Extend the signal by mirroring
    T = save_T
    f_mirror = np.empty(shape=2*T, dtype=np.float64)
    f_mirror[:T//2]       = np.flip(signal[:T//2])
    f_mirror[T//2:3*T//2] = signal
    f_mirror[3*T//2:]     = np.flip(signal[T//2:])
    f = f_mirror

    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    t = np.arange(1, T+1, 1) / T

    # Spectral Domain discretization
    freqs = t - 0.5 - 1/T

    # Maximun number of iterations (if not converged yet, then it won't anyway)
    N = 500

    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat             = np.fft.fftshift(np.fft.fft(f))
    f_hat_plus        = f_hat
    f_hat_plus[:T//2] = 0

    # Matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros((N, len(freqs), K), dtype=np.complex)

    # Initialization of omega_k
    omega_plus = np.zeros((N, K), dtype=np.float64)
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5/K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.random(K)))
    else:
        omega_plus[0, :] = 0
    
    # If DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # Start with emtpy dual variables
    lambda_hat = np.zeros((N, len(freqs)), dtype=np.complex)

    # Other inits
    eps = np.spacing(np.float64(1.0))  # distance from 1.0 to the next largest double-precision number.
    uDiff = tol + eps  # update step
    n = 0  # loop counter
    sum_uk = 0  # accumulator


    # Main loop for iterative updates
    # -------------------------------

    while (uDiff > tol and n < N): # not converged and below iterations limit
        # Update first mode accumulator
        k = 0
        sum_uk += u_hat_plus[n, :, K-1] - u_hat_plus[n, :, 0]

        # Update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n+1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :]/2) \
                                / (1 + Alpha[k] * (freqs - omega_plus[n, k])**2)

        # Update first omega if not held at 0
        if not DC:
            omega_plus[n+1, k] = ((freqs[T//2:T]) * np.matrix(np.abs(u_hat_plus[n+1, T//2:T, k])**2).H) \
                                 / np.sum(np.abs(u_hat_plus[n+1, T//2:T, k])**2)

        # Update of any other mode
        for k in range(1, K):
            # Accumulator
            sum_uk += u_hat_plus[n+1, :, k-1] - u_hat_plus[n, :, k]

            # mode spectrum
            u_hat_plus[n+1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :]/2) \
                                    / (1 + Alpha[k] * (freqs - omega_plus[n, k])**2)

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
            uDiff += 1/T * \
                     sum((u_hat_plus[n, :, i] - u_hat_plus[n-1, :, i]) * \
                         (u_hat_plus[n, :, i] - u_hat_plus[n-1, :, i]).transpose())

        uDiff = np.abs(uDiff)

    # Postprocessing and cleanup
    # --------------------------

    # Discard empty space if converged early
    N = min(N, n+1)
    omega = omega_plus[:N, :]

    # Signal reconstruction
    u_hat              = np.zeros((T, K), dtype=np.complex)
    u_hat[T//2:T, :]   = np.squeeze(u_hat_plus[N-1, T//2:T, :])
    u_hat[1:T//2+1, :] = np.flip(np.squeeze(u_hat_plus[N-1, T//2:T, :].conjugate()))
    u_hat[0, :]        = u_hat[-1, :].conjugate()

    u = np.zeros((K, len(t)), dtype=np.float64)

    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # Remove mirror part
    u = u[:, T//4:3*T//4]

    # Recompute spectrum
    del u_hat
    u_hat = np.empty(shape=(u.shape[1], K), dtype=np.complex)
    for k in range(K):
        u_hat[:, k] = np.matrix(np.fft.fftshift(np.fft.fft(u[k, :]))).H.flatten()


    return u, u_hat, omega
