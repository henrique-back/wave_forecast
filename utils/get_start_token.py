import torch
import numpy as np
from utils import compute_hs

def get_start_token(src, target, freqs, device):
    last_spectrum = src[:, -1, :, 0]  # Select spectrum channel
    if target == 'hs':
        hs = compute_hs(last_spectrum.cpu().numpy(), freqs.cpu().numpy())
        return torch.from_numpy(hs).to(device).float().unsqueeze(-1)
    else:
        return last_spectrum
