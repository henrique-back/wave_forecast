import numpy as np

def compute_hs(density_np, freqs):
    m0 = np.trapezoid(density_np, freqs, axis=1)  # (time,)
    hs = 4 * np.sqrt(m0)  # (time,)
    return hs