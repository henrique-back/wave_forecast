"""
Plot every loss-ablation phase's predicted spectrum against true and
persistence, for one representative test-set sample per characteristic sea
state (wind-sea-only, swell-only, multimodal) — the visual counterpart to
scripts/compare_ablation_phases.py's numeric panel, and structurally the
same PDF-overlay style scripts/plot_cdf_wasserstein.py already uses (true
solid black, persistence dashed, prediction(s) in color), just with one
line per PHASE instead of one line per model.

Representative-sample selection: every test sample's TRUE final-step shape
is classified via utils.spectral_partitioning.find_peak_windows +
classify_partition (same Portilla/Violante-Carvalho criteria used
throughout this ablation's own metrics) into:
    'wind_sea'   exactly one significant peak, gamma* > 1
    'swell'      exactly one significant peak, gamma* <= 1
    'multimodal' 2+ significant peaks (any label mix)
Within each bucket, the sample closest to the bucket's MEDIAN true Tm02
is picked as "characteristic" — Tm02 (not Hs/peak height) because it's
scale-invariant (see nn/evaluate.py's Tm02 note), so it's meaningful
directly on a unit-area SHAPE spectrum without needing freq_means, and it's
a criterion that doesn't depend on any one phase's predictions (unlike
scripts/plot_cdf_wasserstein.py's own "median Wasserstein error" pick,
which only makes sense for a single model).

All 5 phases' checkpoints share one pinned architecture (scripts.
ablate_loss.PINNED_CONFIG) and one deterministic test split, so their
predictions are directly overlayable on the same axes.

GPU-heavy (5 autoregressive test-set passes) — must run via Slurm, not
directly (see utils.require_slurm).

Usage:
    python scripts/plot_ablation_spectra.py
    python scripts/plot_ablation_spectra.py --out results/lossablation_spectra_v2.png
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import require_slurm
require_slurm("scripts/plot_ablation_spectra.py")

import matplotlib
matplotlib.use("Agg")  # headless — this always runs via Slurm, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nn import WaveHeightBaselineNN, evaluate
from nn.optimization import _prepare_dataloaders
from utils import get_device, compute_bulk_params
from utils.spectral_partitioning import find_peak_windows, classify_partition

from scripts.ablate_loss import (BUOY_ID, PINNED_CONFIG, TARGET, CHANNEL_SET, AUX_SET,
                                  LEAD_TIME_HOURS, STUDY_VERSION, PHASE_N_TRIALS, _phase_dir)

PHASES = list(PHASE_N_TRIALS)
PHASE_COLORS = {"baseline": "C0", "kl": "C1", "wasserstein": "C2", "peak": "C3", "combined": "C4"}
CATEGORIES = ["wind_sea", "swell", "multimodal"]
CATEGORY_TITLES = {"wind_sea": "Wind-sea (unimodal)", "swell": "Swell (unimodal)",
                    "multimodal": "Multimodal (2+ peaks)"}


def classify_sample(freqs_np, true_spec):
    """True final-step physical shape -> 'wind_sea' | 'swell' | 'multimodal' | None
    (None = no significant peak at all — excluded from every bucket)."""
    windows = find_peak_windows(freqs_np, true_spec)
    if len(windows) == 0:
        return None
    if len(windows) >= 2:
        return "multimodal"
    peak_idx, _, _ = windows[0]
    label = classify_partition(fp=freqs_np[peak_idx], S_obs_at_fp=true_spec[peak_idx])
    return label  # 'wind_sea' or 'swell'


def pick_representative(indices, tm02_values):
    """Index (into the ORIGINAL test-set array) of the sample whose true
    Tm02 is closest to this bucket's median — a "typical" example by a
    criterion that doesn't depend on any one phase's predictions."""
    vals = tm02_values[indices]
    median = np.median(vals)
    local_pos = int(np.argmin(np.abs(vals - median)))
    return indices[local_pos]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str,
                         default=str(Path(__file__).parent.parent / "results"
                                     / f"lossablation_spectra_{STUDY_VERSION.split('_')[-1]}.png"),
                         help="Output image path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "buoy_data" / BUOY_ID / "processed_data.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found — run scripts/data_processing.py first.")
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(file_path)

    device = get_device()
    print(f"Running on device: {device}")

    # One shared test_loader — same pinned config every phase's search used
    # (see scripts/evaluate_ablation_phases.py's identical reasoning).
    _, _, test_loader, _, _, num_freqs, num_channels, num_aux_channels = _prepare_dataloaders(
        density, alpha_1, alpha_2, r_1, r_2,
        PINNED_CONFIG["seq_len"], LEAD_TIME_HOURS, PINNED_CONFIG["batch_size"], TARGET,
        shuffle_seed=0, wind=wind, channel_set=CHANNEL_SET, aux_set=AUX_SET,
    )
    embed_dim = PINNED_CONFIG["head_dim"] * PINNED_CONFIG["nhead"]

    freqs_np = None
    true_final = None
    pers_final = None
    pred_final_by_phase = {}

    for phase in PHASES:
        ckpt_path = _phase_dir(phase) / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[{phase}] no best_model.pt at {ckpt_path} — skipping.")
            continue

        print(f"[{phase}] loading checkpoint and running inference on the test set...")
        # map_location='cpu': FreqDimEmbedding builds a fixed sinusoidal
        # buffer from `freqs` at construction time (before .to(device)) —
        # see scripts/evaluate_ablation_phases.py's identical fix/comment.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

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

        _, (y_pred_all, y_true_all, y_pers_all) = evaluate(
            model, test_loader, device, ckpt["freqs"], lead_time=ckpt["lead_time_steps"],
            freq_means=ckpt["freq_means"], shape_means=ckpt["shape_means"],
            return_arrays=True,
        )

        # LOG-space for target='shape' (see nn/evaluate.py's docstring NOTE)
        # -> exp() back to physical unit-area shape, final step only.
        pred_final_by_phase[phase] = np.exp(y_pred_all[:, -1, :].numpy())

        if freqs_np is None:
            freqs_np = ckpt["freqs"].numpy()
            true_final = np.exp(y_true_all[:, -1, :].numpy())
            pers_final = np.exp(y_pers_all[:, -1, :].numpy())

    if not pred_final_by_phase:
        raise RuntimeError("No phase had a best_model.pt to evaluate — nothing to plot.")

    n = true_final.shape[0]
    labels = [classify_sample(freqs_np, true_final[i]) for i in range(n)]
    _, tm02_all = compute_bulk_params(true_final, freqs_np)  # Hs discarded, meaningless on shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, category in zip(axes, CATEGORIES):
        indices = np.array([i for i in range(n) if labels[i] == category])
        if indices.size == 0:
            ax.set_title(f"{CATEGORY_TITLES[category]} — no test sample found")
            ax.axis("off")
            continue
        idx = pick_representative(indices, tm02_all)

        ax.plot(freqs_np, true_final[idx], "k-", linewidth=2.5, label="True")
        ax.plot(freqs_np, pers_final[idx], color="gray", linestyle="--", linewidth=1.5,
                label="Persistence")
        for phase in PHASES:
            if phase not in pred_final_by_phase:
                continue
            ax.plot(freqs_np, pred_final_by_phase[phase][idx], color=PHASE_COLORS[phase],
                    linewidth=1.8, label=phase)

        ax.set_title(f"{CATEGORY_TITLES[category]} — sample {idx} (Tm02={tm02_all[idx]:.2f}s)")
        ax.set_xlabel("Frequency (Hz)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Shape E(f)/m₀")
    axes[0].legend(fontsize=9, loc="upper right")

    fig.suptitle(f"Loss-ablation phases vs. true/persistence — target={TARGET}, "
                 f"lead={LEAD_TIME_HOURS}h, {STUDY_VERSION}")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot → {out_path}")


if __name__ == "__main__":
    main()
