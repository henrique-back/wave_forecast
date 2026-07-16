import numpy as np

def compute_hs_from_density(density_batch, freqs):
    """
    Compute significant wave height Hs from a batch of spectra using trapezoidal integration.

    Parameters
    ----------
    density_batch : np.ndarray
        Spectral density, shape (batch, lead_time, num_freqs) 
        or (n_samples, num_freqs).
    freqs : np.ndarray
        Frequency array, shape (num_freqs,).

    Returns
    -------
    hs : np.ndarray
        Significant wave height, shape (batch, lead_time) 
        or (n_samples,).
    """
    # Handle both 2D (time, freq) and 3D (batch, lead_time, freq) inputs
    if density_batch.ndim == 2:
        # (time, num_freqs)
        m0 = np.trapezoid(density_batch, freqs, axis=1)  # (time,)
    elif density_batch.ndim == 3:
        # (batch, lead_time, num_freqs)
        m0 = np.trapezoid(density_batch, freqs, axis=2)  # (batch, lead_time)
    else:
        raise ValueError("density_batch must be 2D or 3D array")

    hs = 4 * np.sqrt(m0)
    return hs


def compute_bulk_params(density_batch, freqs):
    """
    Compute Hs and Tm02 from a batch of spectra.

    Parameters
    ----------
    density_batch : np.ndarray
        Spectral density, shape (batch, lead_time, num_freqs)
        or (n_samples, num_freqs).
    freqs : np.ndarray
        Frequency array, shape (num_freqs,).

    Returns
    -------
    hs   : np.ndarray  — 4√m₀,  same leading shape as density_batch sans freq axis
    tm02 : np.ndarray  — √(m₀/m₂)
    """
    freq_axis = density_batch.ndim - 1  # last axis is always frequency

    m0 = np.trapezoid(density_batch,               freqs, axis=freq_axis).clip(min=0.0)
    m2 = np.trapezoid(density_batch * freqs ** 2,  freqs, axis=freq_axis).clip(min=1e-12)

    hs   = 4.0 * np.sqrt(m0)
    tm02 = np.sqrt(m0.clip(min=0.0) / m2)
    return hs, tm02


def trapz_weights(freqs):
    """
    Per-bin trapezoidal integration weights for a frequency grid, normalised
    to sum to 1.

    The buoy's frequency grid is log-spaced (dense near 0.02 Hz, coarse near
    0.485 Hz), so a plain arithmetic mean over bins over-represents the
    low-frequency region relative to its actual contribution to a physical
    integral. weights * values summed approximates
    ∫ f(x) dx / (freqs[-1] - freqs[0]) — i.e. a frequency-weighted mean
    consistent with the trapezoidal rule used elsewhere (compute_bulk_params,
    compute_shape) for spectral moments.

    Parameters
    ----------
    freqs : np.ndarray, shape (num_freqs,)

    Returns
    -------
    weights : np.ndarray, shape (num_freqs,), sums to 1.
    """
    freqs = np.asarray(freqs, dtype=float)
    w = np.empty_like(freqs)
    w[0] = (freqs[1] - freqs[0]) / 2
    w[-1] = (freqs[-1] - freqs[-2]) / 2
    w[1:-1] = (freqs[2:] - freqs[:-2]) / 2
    return w / w.sum()


def compute_shape(density_batch, freqs, m0_threshold=1e-12):
    """
    Normalize a batch of spectra to unit-area shape: shape(f) = E(f) / m0.

    This decouples spectral shape from energy magnitude — the quantity a
    'shape'-target model is trained to predict, paired separately with an
    'hs'-target model that forecasts magnitude (see CLAUDE.md's discussion
    of the shape/magnitude model split).

    Parameters
    ----------
    density_batch : np.ndarray
        Spectral density, shape (n_samples, num_freqs) or
        (batch, lead_time, num_freqs) — same convention as
        compute_hs_from_density / compute_bulk_params.
    freqs : np.ndarray
        Frequency array, shape (num_freqs,).
    m0_threshold : float
        m0 is clipped to this minimum before dividing, so near-zero-energy
        spectra (calm seas, missing records) don't produce inf/nan.

    Returns
    -------
    shape : np.ndarray, same shape as density_batch — integrates to 1 over freqs.
    """
    freq_axis = density_batch.ndim - 1
    m0 = np.trapezoid(density_batch, freqs, axis=freq_axis).clip(min=m0_threshold)
    return density_batch / np.expand_dims(m0, axis=freq_axis)