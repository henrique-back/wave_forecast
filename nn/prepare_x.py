import numpy as np
import torch

def prepare_X(channels, seq_length=24, lead_time=1):
    """
    Stack frequency-resolved input channels into shape (samples, seq_len, freqs, channels)

    Parameters:
    - channels: list of pandas DataFrames (time x freqs), all matching shape/index.
      density must be first — utils/get_start_token.py reads channel 0 as density.

    Returns:
    - X: torch.FloatTensor of shape (samples, seq_length, freqs, len(channels))
    """
    arrays = [c.values.astype(np.float32) for c in channels]

    num_timesteps, num_freqs = arrays[0].shape
    num_samples = num_timesteps - seq_length - lead_time + 1

    X = np.zeros((num_samples, seq_length, num_freqs, len(arrays)), dtype=np.float32)

    for i in range(num_samples):
        for c, arr in enumerate(arrays):
            X[i, :, :, c] = arr[i:i+seq_length]

    return torch.from_numpy(X)
