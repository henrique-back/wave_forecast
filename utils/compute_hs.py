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