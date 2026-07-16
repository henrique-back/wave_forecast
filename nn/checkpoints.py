"""Shared checkpoint lookup/loading convention.

Used by scripts/infer.py, scripts/compare_versions.py, and nn/spectrum_eval.py
— anything that loads a scripts/train.py checkpoint by experiment/target/lead
and needs to rebuild the WaveHeightBaselineNN it came from.
"""
import torch

from nn.transformer import WaveHeightBaselineNN
from nn.channels import CHANNEL_SETS, AUX_CHANNEL_SETS


def find_checkpoint(project_root, experiment, target, deltat, lead, seed):
    """Return (ckpt, results_folder) for the first checkpoint that exists.

    Prefers scripts/train.py's final retrain (final_model_seed{N}.pt) over the
    raw Optuna best_model.pt, and prefers the deltat-namespaced results path
    (current scripts/optimize.py convention) over the older path some earlier
    experiments were saved under (results/{experiment}/{target}/lead_{N}h/,
    no deltat_{N} level) — e.g. results/hs_shape_v5 predates that convention.
    """
    candidate_folders = [
        project_root / 'results' / experiment / target / f'deltat_{deltat}' / f'lead_{lead}h',
        project_root / 'results' / experiment / target / f'lead_{lead}h',
    ]
    candidate_files = [f'final_model_seed{seed}.pt', 'best_model.pt']

    for folder in candidate_folders:
        for fname in candidate_files:
            ckpt_path = folder / fname
            if ckpt_path.exists():
                print(f"Loaded {target!r} checkpoint from {ckpt_path}")
                return torch.load(ckpt_path, map_location='cpu', weights_only=False), folder

    raise FileNotFoundError(
        f"No checkpoint found for experiment={experiment!r} target={target!r} "
        f"deltat={deltat} lead={lead}h seed={seed}. Looked for "
        f"{candidate_files} under {[str(f) for f in candidate_folders]}."
    )


def build_model(ckpt, freqs, device, channel_set='full', aux_set='none'):
    params = ckpt['params']
    embed_dim = params['head_dim'] * params['nhead']
    model = WaveHeightBaselineNN(
        num_freqs=len(freqs),
        freqs=freqs,
        target=ckpt['target'],
        num_channels=len(CHANNEL_SETS[channel_set]),
        num_aux_channels=len(AUX_CHANNEL_SETS[aux_set]),
        dropout=params['dropout'],
        nhead=params['nhead'],
        num_encoder_layers=params['num_encoder_layers'],
        num_decoder_layers=params['num_decoder_layers'],
        embed_dim=embed_dim,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model
