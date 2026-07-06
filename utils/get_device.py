import os
import torch


def get_device() -> str:
    """Select the best available compute device: CUDA > MPS > CPU.

    `torch.backends.mps.is_available()` can report True on a machine/torch
    build where the backend is present but not actually usable (e.g. certain
    macOS/Rosetta combinations), so it is verified here with a real op
    instead of trusting the flag alone. If the smoke test fails, falls back
    to CPU instead of crashing later mid-training.

    Also sets PYTORCH_ENABLE_MPS_FALLBACK=1 when MPS is selected: some ops
    are not yet implemented for MPS and raise NotImplementedError — this env
    var makes PyTorch silently run just that op on CPU instead of crashing
    the whole run.
    """
    if torch.cuda.is_available():
        return 'cuda'

    if torch.backends.mps.is_available():
        try:
            (torch.ones(1, device='mps') + 1).cpu()
        except Exception as e:
            print(f'MPS reported available but failed a smoke test ({e}); falling back to CPU.')
            return 'cpu'
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        return 'mps'

    return 'cpu'


def empty_cache(device) -> None:
    """Free cached allocator memory for whichever backend `device` refers to."""
    device = str(device)
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()
