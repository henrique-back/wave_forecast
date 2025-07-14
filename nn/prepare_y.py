import numpy as np
import torch
from utils import compute_hs

def prepare_y(density, seq_length=24, lead_time=1, target="hs"):
    """
    Prepare target values for time series forecasting.

    Parameters:
    - density: pandas DataFrame (time x frequencies), column names must be float frequencies
    - seq_length: input window length
    - lead_time: steps ahead to predict
    - target: 'hs' for significant wave height, 'density' for full spectrum

    Returns:
    - torch.FloatTensor:
        - shape (samples, lead_time, 1) if target == 'hs'
        - shape (samples, lead_time, num_freqs) if target == 'density'
    """
    try:
        freqs = np.array([float(f) for f in density.columns])
    except:
        raise ValueError("Column names must be float frequencies (e.g., '0.034')")

    density_np = density.values.astype(np.float32)  # (time, freqs)
    num_timesteps, num_freqs = density_np.shape

    num_samples = num_timesteps - seq_length - lead_time + 1

    if target == "hs":
        hs = compute_hs(density_np, freqs)  # (time,)
        y = np.zeros((num_samples, lead_time, 1), dtype=np.float32)
        for i in range(num_samples):
            y[i, :, 0] = hs[i + seq_length : i + seq_length + lead_time]
        return torch.from_numpy(y)

    elif target == "density":
        y = np.zeros((num_samples, lead_time, num_freqs), dtype=np.float32)
        for i in range(num_samples):
            y[i, :, :] = density_np[i + seq_length : i + seq_length + lead_time, :]
        return torch.from_numpy(y)

    else:
        raise ValueError("Invalid target type. Choose 'hs' or 'density'.")