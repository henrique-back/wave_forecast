"""
Dynamic Mode Decomposition (DMD) features from the encoder's input window.

Rather than asking the Transformer to infer implicitly from raw historical
spectra whether the current swell/wind-sea systems are growing or decaying,
DMD fits a linear operator A such that x_{t+1} ~= A x_t across a sample's
seq_len window of (already-windowed, already-normalized) density spectra,
then decomposes A into modes with complex eigenvalues -- each eigenvalue's
magnitude gives a growth/decay rate, its phase gives an oscillation
frequency. These per-mode (growth_rate, frequency, amplitude) triples are
exposed as a new 'dmd' aux_set (nn/channels.py), broadcast across seq_len to
match nn/prepare_aux.py's (samples, seq_len, channels) output contract --
see that module's docstring for why DMD needs a *different* preparation
function rather than reusing prepare_aux itself (DMD needs each sample's
already-windowed history FIRST, to fit DMD on that sample's own history,
whereas prepare_aux windows an already-fully-computed per-timestep series).

Normalized (not physical) density is fine as DMD's input: DMD eigenvalues
are invariant to a fixed per-bin linear rescaling (a similarity transform:
if x_norm = D^-1 x_phys for a fixed diagonal D, then A_norm = D^-1 A_phys D,
which has the SAME eigenvalues as A_phys) -- only mode amplitude becomes
"relative to normalized units," still a meaningful feature.
"""
import numpy as np

DEFAULT_N_MODES = 4  # MUST stay in sync with nn.channels._DMD_COLUMNS


def compute_dmd_features(density_window_X, n_modes=DEFAULT_N_MODES, dt=1.0):
    """
    Parameters
    ----------
    density_window_X : np.ndarray, shape (num_samples, seq_len, num_freqs)
        The already-windowed, already-normalized density channel (e.g.
        train_X[..., 0].numpy() from nn/prepare_x.py::prepare_X).
    n_modes : int
        Number of dominant modes to keep per sample (sorted by amplitude).
    dt : float
        Time step between snapshots, in the same units the returned growth
        rate/frequency are expressed in (this project's data is hourly, so
        dt=1.0 gives per-hour growth rate and radians/hour frequency).

    Returns
    -------
    np.ndarray, shape (num_samples, 3*n_modes), float32 — per sample, flat
    [growth_0, freq_0, amp_0, growth_1, freq_1, amp_1, ...], modes sorted by
    amplitude descending, zero-padded if fewer than n_modes valid modes are
    found. Real input data gives complex eigenvalues in conjugate pairs (a
    real decaying mode has one; an oscillating mode has two, with matching
    growth rate and +/-frequency) -- only Im(omega) >= 0 modes are kept, so
    a conjugate pair contributes ONE physical mode, not two.
    """
    window = np.asarray(density_window_X, dtype=np.float64)
    n_samples, seq_len, num_freqs = window.shape

    # State vectors are columns: (num_samples, num_freqs, seq_len-1).
    X1 = window[:, :-1, :].transpose(0, 2, 1)
    X2 = window[:, 1:, :].transpose(0, 2, 1)
    T = seq_len - 1

    # Extra headroom over n_modes (a real candidate pool to select the best
    # n_modes from after conjugate-pair filtering, not exactly n_modes
    # forced choices).
    r = max(1, min(2 * n_modes, num_freqs, T))

    U, S, Vh = np.linalg.svd(X1, full_matrices=False)
    U_r = U[..., :, :r]                                    # (N, F, r)
    S_r = S[..., :r]                                       # (N, r)
    inv_S_r = 1.0 / np.clip(S_r, 1e-10, None)
    V_r = Vh[..., :r, :].conj().transpose(0, 2, 1)          # (N, T, r)

    # Reduced operator A_tilde = U_r^H @ X2 @ V_r @ diag(1/S_r).
    A_tilde = U_r.conj().transpose(0, 2, 1) @ X2 @ V_r      # (N, r, r)
    A_tilde = A_tilde * inv_S_r[:, None, :]

    eigvals, W = np.linalg.eig(A_tilde)                     # (N, r), (N, r, r)

    # DMD modes Phi = X2 @ V_r @ diag(1/S_r) @ W.
    Phi = (X2 @ V_r) * inv_S_r[:, None, :]                  # (N, F, r)
    Phi = Phi @ W                                           # (N, F, r), complex

    x1 = window[:, 0, :]                                    # (N, F)
    b = np.linalg.pinv(Phi) @ x1[:, :, None]                # (N, r, 1)
    b = b[:, :, 0]                                           # (N, r)

    # Continuous-time eigenvalues: clip magnitude away from 0 before log()
    # (a near-zero/degenerate mode would otherwise give -inf).
    mag = np.clip(np.abs(eigvals), 1e-10, None)
    phase = np.angle(eigvals)
    omega = (np.log(mag) + 1j * phase) / dt                 # (N, r)

    growth_rate = omega.real
    ang_freq = omega.imag
    amp_mag = np.abs(b)

    out = np.zeros((n_samples, 3 * n_modes), dtype=np.float64)
    for i in range(n_samples):
        keep = np.where(ang_freq[i] >= 0)[0]  # drop conjugate duplicates
        order = keep[np.argsort(-amp_mag[i, keep])][:n_modes]
        m = len(order)
        out[i, 0:3 * m:3] = growth_rate[i, order]
        out[i, 1:3 * m:3] = ang_freq[i, order]
        out[i, 2:3 * m:3] = amp_mag[i, order]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
