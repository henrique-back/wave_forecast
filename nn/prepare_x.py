import numpy as np
import torch

def prepare_X(density, alpha1, alpha2, r1, seq_length=24, lead_time=1):
    """
    Stack input features into shape (samples, seq_len, freqs, channels)
    
    Parameters:
    - density, alpha1, df_alpha2, df_r1: pandas DataFrames with matching shape and index
    
    Returns:
    - X: torch.FloatTensor of shape (samples, seq_length, freqs, 4)
    """
    # Convert all to numpy arrays
    density = density.values.astype(np.float32)
    alpha1  = alpha1.values.astype(np.float32)
    alpha2  = alpha2.values.astype(np.float32)
    r1      = r1.values.astype(np.float32)

    num_timesteps, num_freqs = density.shape
    num_samples = num_timesteps - seq_length - lead_time + 1

    # Stack the time-series into sequences
    X = np.zeros((num_samples, seq_length, num_freqs, 4), dtype=np.float32)

    for i in range(num_samples):
        X[i, :, :, 0] = density[i:i+seq_length]
        X[i, :, :, 1] = alpha1[i:i+seq_length]
        X[i, :, :, 2] = alpha2[i:i+seq_length]
        X[i, :, :, 3] = r1[i:i+seq_length]

    return torch.from_numpy(X)