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