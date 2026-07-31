import torch
import numpy as np
from utils import compute_hs, compute_shape, to_log_space


def get_start_token(src, target, freqs, device, freq_means=None, shape_means=None):
    """Return the decoder start token for autoregressive inference.

    For the 'hs' target the start token is the significant wave height
    derived from the last observed spectrum.  The spectrum is stored in
    scale-normalised form (Ẽ = E / μ(f)); denormalisation E = Ẽ * μ(f)
    must be applied before the spectral integration so that the result is
    in physical metres.  freq_means (shape: num_freqs) carries μ(f).

    For the 'density' target the start token is log(E_phys) of the last
    observed spectrum — denormalised via freq_means then log-transformed
    (floored per utils.to_log_space) — since the decoder now operates in
    log-spectral-energy space throughout (see nn/training_loop.py).

    For the 'shape' target the start token is log(E(f)/m0) of the last
    observed spectrum — the unit-area shape is computed after denormalising
    by freq_means (m0, and therefore the shape, computed on the
    scale-normalised spectrum would be distorted by the per-bin μ(f)
    scaling rather than reflecting the true physical shape), then
    log-transformed using shape_means as the floor reference.

    Parameters
    ----------
    src        : torch.Tensor, shape (batch, seq_len, num_freqs, channels)
    target     : str, 'hs', 'density', or 'shape'
    freqs      : torch.Tensor, shape (num_freqs,)
    device     : str or torch.device
    freq_means : torch.Tensor | None, shape (num_freqs,); per-frequency
                 training mean μ(f) of the physical density. Required for
                 target in ('hs', 'density', 'shape').
    shape_means : torch.Tensor | None, shape (num_freqs,); per-frequency
                 training mean of the physical unit-area shape target.
                 Required for target == 'shape'.
    """
    # Channel 0 of the last encoder step is the (normalised) spectral density
    last_spectrum = src[:, -1, :, 0]   # (batch, num_freqs), normalised

    if target == 'hs':
        if freq_means is None:
            raise ValueError("freq_means is required for target='hs'")
        # Denormalise: E_phys = Ẽ * μ(f)
        spectrum_phys = last_spectrum * freq_means.to(device)
        hs = compute_hs(spectrum_phys.cpu().numpy(), freqs.cpu().numpy())
        return torch.from_numpy(hs).to(device).float().unsqueeze(-1)

    elif target == 'density':
        if freq_means is None:
            raise ValueError("freq_means is required for target='density'")
        fm = freq_means.to(device)
        spectrum_phys = last_spectrum * fm
        return to_log_space(spectrum_phys, fm)

    elif target == 'shape':
        if freq_means is None or shape_means is None:
            raise ValueError("freq_means and shape_means are both required for target='shape'")
        spectrum_phys = last_spectrum * freq_means.to(device)
        shape = compute_shape(spectrum_phys.cpu().numpy(), freqs.cpu().numpy())
        shape_t = torch.from_numpy(shape).to(device).float()
        return to_log_space(shape_t, shape_means.to(device))

    else:
        raise ValueError(f"Unknown target {target!r}. Choose 'hs', 'density', or 'shape'.")
