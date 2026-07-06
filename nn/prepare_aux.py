import numpy as np
import torch

def prepare_aux(channels, num_timesteps, seq_length=24, lead_time=1):
    """
    Stack scalar-per-timestep auxiliary channels (e.g. wind_u/wind_v) into
    shape (samples, seq_len, num_aux). Unlike prepare_X, these are not
    frequency-resolved — no freq axis.

    Parameters:
    - channels: list of 1-D pandas Series (e.g. [wind_u, wind_v]); may be empty.
    - num_timesteps: length of the reference time axis (e.g. len(density)),
      passed explicitly so an empty channel list still yields a correctly
      shaped (num_samples, seq_len, 0) tensor.

    Returns:
    - aux: torch.FloatTensor of shape (samples, seq_length, len(channels))
    """
    num_samples = num_timesteps - seq_length - lead_time + 1
    aux = np.zeros((num_samples, seq_length, len(channels)), dtype=np.float32)

    for c, series in enumerate(channels):
        values = series.values.astype(np.float32)
        for i in range(num_samples):
            aux[i, :, c] = values[i:i+seq_length]

    return torch.from_numpy(aux)
