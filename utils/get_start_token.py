import torch
import numpy as np
from utils import compute_hs, compute_shape


def get_start_token(src, target, freqs, device, freq_means=None):
    """Return the decoder start token for autoregressive inference.

    For the 'hs' target the start token is the significant wave height
    derived from the last observed spectrum.  The spectrum is stored in
    scale-normalised form (Ẽ = E / μ(f)); denormalisation E = Ẽ * μ(f)
    must be applied before the spectral integration so that the result is
    in physical metres.  freq_means (shape: num_freqs) carries μ(f).

    For the 'shape' target the start token is the unit-area shape E(f)/m0
    of the last observed spectrum — also requires denormalising by
    freq_means first, since m0 (and therefore the shape) computed on the
    scale-normalised spectrum would be distorted by the per-bin μ(f)
    scaling rather than reflecting the true physical shape.

    For the 'density' target the start token is the last observed
    normalised spectrum itself — the model operates in normalised space
    and the persistence baseline must be in the same space.

    Parameters
    ----------
    src        : torch.Tensor, shape (batch, seq_len, num_freqs, channels)
    target     : str, 'hs', 'density', or 'shape'
    freqs      : torch.Tensor, shape (num_freqs,)
    device     : str or torch.device
    freq_means : torch.Tensor | None, shape (num_freqs,); per-frequency
                 training mean μ(f).  Required for physically correct Hs
                 when target == 'hs'; if None the normalised spectrum is
                 used and the result is not in physical metres.
    """
    # Channel 0 of the last encoder step is the (normalised) spectral density
    last_spectrum = src[:, -1, :, 0]   # (batch, num_freqs), normalised

    if target in ('hs', 'shape'):
        if freq_means is not None:
            # Denormalise: E_phys = Ẽ * μ(f)
            spectrum_phys = last_spectrum * freq_means.to(device)
        else:
            spectrum_phys = last_spectrum   # fallback — not in physical metres

        if target == 'hs':
            hs = compute_hs(spectrum_phys.cpu().numpy(), freqs.cpu().numpy())
            return torch.from_numpy(hs).to(device).float().unsqueeze(-1)
        else:
            shape = compute_shape(spectrum_phys.cpu().numpy(), freqs.cpu().numpy())
            return torch.from_numpy(shape).to(device).float()
    else:
        # density target: return normalised spectrum as-is
        return last_spectrum
