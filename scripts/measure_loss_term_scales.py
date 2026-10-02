"""
Measure the raw magnitude of each composite-loss term, to regenerate
nn/optimization.py::LOSS_TERM_REFERENCE.

nn/training_loop.py::train_one_epoch sums four terms raw:

    L = base_w * MSE_log + w2_w * W2 + kl_w * D_KL + peak_w * L_peak

and nothing rescales them, yet they are expressed in four unrelated units —
log-space squared error, Hz, nats, and squared physical E². Measured on buoy
32012 they span about 4.5 orders of magnitude, so a weight's numeric value
says nothing on its own about how much that term actually contributes. The
production search samples the three weights independently, which is how v13
ended up with a space whose MEDIAN draw was ~350:1 peak-dominated (decision
033).

LOSS_TERM_REFERENCE fixes that by expressing each weight as a CONTRIBUTION —
weight = contribution / reference_magnitude — so a sampled value means "this
term is worth this much of the loss". That only works while the reference
magnitudes are right, so re-run this and update the constant whenever
something changes what a term measures: the peak detector (decision 030
moved the median window 5 -> 16 bins, which changes L_peak), the target, the
normalisation, or the frequency grid.

Reports the median/min/max of each term over a few TRAINING batches under
pure teacher forcing, matching how train_one_epoch computes them at
tf_ratio=1.0. The median is what LOSS_TERM_REFERENCE should carry — these are
scale references, not tight estimates, and nothing downstream depends on more
than their order of magnitude.

CPU-only and read-only (no training, no writes), so it does NOT go through
Slurm — same convention as scripts/compare_versions.py.

Usage:
    python scripts/measure_loss_term_scales.py
    python scripts/measure_loss_term_scales.py --phase baseline --n-batches 12
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch

from nn import WaveHeightBaselineNN
from nn.optimization import _prepare_dataloaders, LOSS_TERM_REFERENCE
from nn.training_loop import _peak_windows_for_batch
from utils import RMSELoss, trapz_weights, to_log_space, get_start_token
from utils.loss import (SpectralWassersteinLoss, SpectralKLDivergenceLoss,
                         SoftPeakHeightLoss)

from scripts.ablate_loss import (BUOY_ID, PINNED_CONFIG, TARGET, CHANNEL_SET, AUX_SET,
                                  LEAD_TIME_HOURS, MAX_PEAKS, _phase_dir)

# Key in LOSS_TERM_REFERENCE -> human label.
TERMS = {"base": "base (trapz-weighted MSE, log-space)",
         "w2": "W2 (Hz)",
         "kl": "KL (nats)",
         "peak": "peak (E^2)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="kl",
                         help="Loss-ablation phase whose best_model.pt to measure with "
                              "(default 'kl' — the checkpoint the committed "
                              "LOSS_TERM_REFERENCE was derived from). Any phase works: "
                              "the terms are evaluated on the same data with the same "
                              "pinned architecture, so the reference magnitudes barely "
                              "move between them.")
    parser.add_argument("--n-batches", type=int, default=6,
                         help="Training batches to average over (default 6)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "buoy_data" / BUOY_ID / "processed_data.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found — run scripts/data_processing.py first.")
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(file_path)

    ckpt_path = _phase_dir(args.phase) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} — run that ablation phase first, or pass "
            f"--phase for one that has finished.")

    # Pure teacher forcing below, so only the train_loader's batching matters;
    # shuffle_seed=0 matches scripts/evaluate_ablation_phases.py's convention.
    train_loader, _, _, freq_means, shape_means, num_freqs, num_channels, num_aux = \
        _prepare_dataloaders(
            density, alpha_1, alpha_2, r_1, r_2,
            PINNED_CONFIG["seq_len"], LEAD_TIME_HOURS, PINNED_CONFIG["batch_size"], TARGET,
            shuffle_seed=0, wind=wind, channel_set=CHANNEL_SET, aux_set=AUX_SET)

    # map_location='cpu' is required, not stylistic: FreqDimEmbedding builds a
    # fixed sinusoidal buffer from `freqs` at construction time (same note as
    # scripts/evaluate_ablation_phases.py).
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = WaveHeightBaselineNN(
        num_freqs=num_freqs, freqs=ckpt["freqs"], target=TARGET,
        num_channels=num_channels, num_aux_channels=num_aux,
        freq_embed_dropout=PINNED_CONFIG["freq_embed_dropout"],
        embed_dropout=PINNED_CONFIG["embed_dropout"], nhead=PINNED_CONFIG["nhead"],
        num_encoder_layers=PINNED_CONFIG["num_encoder_layers"],
        num_decoder_layers=PINNED_CONFIG["num_decoder_layers"],
        embed_dim=PINNED_CONFIG["head_dim"] * PINNED_CONFIG["nhead"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    freqs = ckpt["freqs"]
    freq_weights = trapz_weights(freqs)
    freqs_np = freqs.numpy().astype(float)
    base_fn, w2_fn = RMSELoss(), SpectralWassersteinLoss()
    kl_fn, peak_fn = SpectralKLDivergenceLoss(), SoftPeakHeightLoss()

    values = {k: [] for k in TERMS}
    with torch.no_grad():
        for i, (src, aux, y_batch) in enumerate(train_loader):
            if i >= args.n_batches:
                break
            # Same log-space conversion and teacher-forced decoder input that
            # train_one_epoch builds at tf_ratio >= 1.0.
            y_batch = to_log_space(y_batch, shape_means)
            start_token = get_start_token(src, TARGET, freqs, "cpu",
                                          freq_means=freq_means, shape_means=shape_means)
            tgt = torch.zeros_like(y_batch)
            tgt[:, 0, :] = start_token
            tgt[:, 1:, :] = y_batch[:, :-1, :]
            y_pred = model(src, tgt, aux=aux)

            left_idx, right_idx, peak_mask = _peak_windows_for_batch(
                y_batch, freqs_np, max_peaks=MAX_PEAKS)

            values["base"].append(
                base_fn(y_pred, y_batch, weights=freq_weights, squared=True).item())
            values["w2"].append(w2_fn(y_pred, y_batch, freqs).item())
            values["kl"].append(kl_fn(y_pred, y_batch, freqs).item())
            peak_term = peak_fn(y_pred, y_batch, freqs, left_idx, right_idx, peak_mask)
            if not torch.isnan(peak_term):   # all-K=0 batch, same guard as train_one_epoch
                values["peak"].append(peak_term.item())

    n_used = len(values["base"])
    print(f"Raw UNWEIGHTED term magnitudes — phase={args.phase!r}, {n_used} training batches, "
          f"target={TARGET}, lead={LEAD_TIME_HOURS}h\n")
    print(f"{'term':<36}{'median':>12}{'min':>12}{'max':>12}{'committed':>12}")
    print("-" * 84)
    measured = {}
    for key, label in TERMS.items():
        arr = np.asarray(values[key], dtype=float)
        if arr.size == 0:
            print(f"{label:<36}{'(no samples)':>48}")
            continue
        measured[key] = float(np.median(arr))
        print(f"{label:<36}{measured[key]:>12.4g}{arr.min():>12.4g}{arr.max():>12.4g}"
              f"{LOSS_TERM_REFERENCE[key]:>12.4g}")

    print("\nRatio of measured median to the committed LOSS_TERM_REFERENCE:")
    drifted = []
    for key in measured:
        ratio = measured[key] / LOSS_TERM_REFERENCE[key]
        flag = "" if 0.5 <= ratio <= 2.0 else "   <-- drifted"
        if flag:
            drifted.append(key)
        print(f"  {key:<6} {ratio:>8.2f}x{flag}")
    if drifted:
        print(f"\n{', '.join(drifted)} moved by more than 2x. Update LOSS_TERM_REFERENCE in "
              f"nn/optimization.py and record why in the decision log — a stale reference "
              f"silently un-centres the search space (decision 033).")
    else:
        print("\nAll within 2x of the committed reference — no update needed.")


if __name__ == "__main__":
    main()
