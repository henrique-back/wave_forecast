import torch

# Fraction of a per-frequency training-mean reference (freq_means for
# 'density', shape_means for 'shape') used as the pre-log floor, so
# torch.log() never sees an exact 0.0 from a genuinely calm/near-zero bin.
LOG_FLOOR_FRACTION = 1e-3


def to_log_space(y_phys, ref_mean, floor_frac=LOG_FLOOR_FRACTION):
    """log(y_phys), floored per-frequency-bin at floor_frac * ref_mean(f).

    Parameters
    ----------
    y_phys   : torch.Tensor, physical-space value (density E(f) or shape
               E(f)/m0), any shape broadcastable against ref_mean's last axis.
    ref_mean : torch.Tensor, shape (num_freqs,) — per-frequency training-mean
               reference (freq_means for 'density', shape_means for 'shape').
    """
    floor = floor_frac * ref_mean.to(device=y_phys.device, dtype=y_phys.dtype)
    return torch.log(torch.maximum(y_phys, floor))
