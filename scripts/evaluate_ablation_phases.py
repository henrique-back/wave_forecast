"""
Recompute the full evaluate() panel — including Tm02_Bias_windsea/_swell,
which scripts/ablate_loss.py's saved trial.user_attrs is missing (see
scripts/compare_ablation_phases.py's module docstring) — for each loss-
ablation phase's WINNING checkpoint, on the held-out TEST set rather than
the validation set best_trial.txt reports. Test, not val, is the correct
split for a final cross-arm comparison: validation was already used to
pick each phase's winning weight (see nn/optimization.py::_compute_val_
score's 'peak_fidelity_SS' docstring), so it's the same "don't judge a
result using data/criteria that already selected it" principle applied to
the train/val/test split instead of the metric choice.

Loads each phase's best_model.pt (architecture rebuilt from
scripts.ablate_loss.PINNED_CONFIG — the same pinned config every phase's
search used, so every checkpoint shares one architecture), runs ONE
autoregressive evaluate(..., compute_peak_metrics=True) pass per phase over
the test split (shared across phases — same data, same architecture,
deterministic 70/15/15 split regardless of shuffle_seed), and writes the
full JSON-safe metrics dict to
results/lossablation_{phase}_{STUDY_VERSION}/shape/lead_{N}h/test_metrics.json.
scripts/compare_ablation_phases.py prefers this file over best_trial.txt's
validation numbers when present.

A phase whose best_model.pt doesn't exist (not yet finished, or failed
before writing one) is skipped with a message rather than raising — safe
to run even if e.g. one phase's job died.

GPU-heavy (up to 5 autoregressive test-set passes) — must run via Slurm,
not directly (see utils.require_slurm).

Usage:
    python scripts/evaluate_ablation_phases.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import require_slurm
require_slurm("scripts/evaluate_ablation_phases.py")

import pandas as pd
import torch

from nn import WaveHeightBaselineNN, evaluate
from nn.optimization import _prepare_dataloaders
from utils import get_device

from scripts.ablate_loss import (BUOY_ID, PINNED_CONFIG, TARGET, CHANNEL_SET, AUX_SET,
                                  LEAD_TIME_HOURS, STUDY_VERSION, PHASE_N_TRIALS, _phase_dir)

PHASES = list(PHASE_N_TRIALS)


def _to_jsonable(obj):
    """Recursively coerce numpy/torch scalars (float32, int64, ...) into
    plain Python types so json.dump doesn't choke on them — evaluate()'s
    metrics dict is built from numpy arithmetic throughout nn/evaluate.py,
    so this isn't an edge case, it's the common case."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "item"):  # numpy/torch scalar
        return obj.item()
    return obj


def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "buoy_data" / BUOY_ID / "processed_data.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found — run scripts/data_processing.py first.")
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(file_path)

    device = get_device()
    print(f"Running on device: {device}")

    # One shared test_loader: same pinned seq_len/batch_size/channel_set/
    # aux_set as every phase's own search used, and test_loader is always
    # shuffle=False with a deterministic 70/15/15 split boundary (see
    # nn/optimization.py::_prepare_dataloaders) — shuffle_seed doesn't
    # affect which samples land in it, so reusing one loader across all 5
    # phases is exactly the same test set each would build individually.
    _, _, test_loader, _, _, num_freqs, num_channels, num_aux_channels = _prepare_dataloaders(
        density, alpha_1, alpha_2, r_1, r_2,
        PINNED_CONFIG["seq_len"], LEAD_TIME_HOURS, PINNED_CONFIG["batch_size"], TARGET,
        shuffle_seed=0, wind=wind, channel_set=CHANNEL_SET, aux_set=AUX_SET,
    )
    embed_dim = PINNED_CONFIG["head_dim"] * PINNED_CONFIG["nhead"]

    for phase in PHASES:
        ckpt_path = _phase_dir(phase) / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[{phase}] no best_model.pt at {ckpt_path} — skipping.")
            continue

        print(f"[{phase}] loading checkpoint and evaluating on the test set...")
        # map_location='cpu' is required, not stylistic: FreqDimEmbedding
        # builds a fixed sinusoidal buffer from `freqs` at construction time,
        # before .to(device) below runs — loading straight to CUDA/MPS raises
        # a cross-device error there.
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        model = WaveHeightBaselineNN(
            num_freqs=num_freqs,
            freqs=ckpt["freqs"],
            target=TARGET,
            num_channels=num_channels,
            num_aux_channels=num_aux_channels,
            freq_embed_dropout=PINNED_CONFIG["freq_embed_dropout"],
            embed_dropout=PINNED_CONFIG["embed_dropout"],
            nhead=PINNED_CONFIG["nhead"],
            num_encoder_layers=PINNED_CONFIG["num_encoder_layers"],
            num_decoder_layers=PINNED_CONFIG["num_decoder_layers"],
            embed_dim=embed_dim,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        test_metrics = evaluate(
            model, test_loader, device, ckpt["freqs"],
            lead_time=ckpt["lead_time_steps"],
            freq_means=ckpt["freq_means"], shape_means=ckpt["shape_means"],
            compute_peak_metrics=True,
        )
        test_metrics["_phase"] = phase
        test_metrics["_trial_params"] = ckpt["params"]
        test_metrics["_base_loss_weight"] = ckpt.get("base_loss_weight")
        test_metrics["_fixed_loss_weights"] = ckpt.get("fixed_loss_weights")

        out_path = _phase_dir(phase) / "test_metrics.json"
        out_path.write_text(json.dumps(_to_jsonable(test_metrics), indent=2))
        print(f"[{phase}] wrote {out_path}")
        print(f"[{phase}] Peak_Height_RelError windsea/swell: "
              f"{test_metrics.get('Peak_Height_RelError_windsea'):.4f} / "
              f"{test_metrics.get('Peak_Height_RelError_swell'):.4f}   "
              f"Tm02_Bias windsea/swell: "
              f"{test_metrics.get('Tm02_Bias_windsea'):.4f} / "
              f"{test_metrics.get('Tm02_Bias_swell'):.4f}")

    print("\nDone. Re-run scripts/compare_ablation_phases.py to see the updated (test-set) panel.")


if __name__ == "__main__":
    main()
