import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups (if any)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Slower, but reproducible
