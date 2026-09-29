"""
Check how often a trained 'shape' model's predicted peak falls outside the
true peak's trough-to-trough window — the case where utils.loss.
SoftPeakHeightLoss (windows fixed by the TRUE spectrum) would score a
correctly-shaped but displaced peak as missing. See
manuscript/decisions/log/028.

For every true partition the loss actually uses (the first MAX_PEAKS
lowest-frequency windows per spectrum, as nn/training_loop.py::
_peak_windows_for_batch keeps), reports:
  - whether the predicted significant peak nearest the true peak lies
    inside the true window, overall / per wind-sea-swell label / for
    narrow windows;
  - the distance (bins) from the true peak to that nearest predicted peak;
  - for displaced partitions, the peak term's relative height error (soft
    max inside the true window) vs. the error against the nearest
    predicted peak — if the displaced predicted peak were an intact copy
    of the true one, the second would be much smaller than the first.

Wind-sea/swell labels are computed on the PHYSICAL density at the true peak
(not the unit-area shape nn/evaluate.py passes to classify_partition), so
they are the labels gamma* is defined for.

Runs autoregressive inference only (no training) on CPU. Output is printed
and written to results/peak_window_check/{EXPERIMENT}_lead_{N}h_{SPLIT}.txt.

Run manually:
    python scripts/check_peak_windows.py
"""
print("Importing packages")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nn import evaluate
from nn.checkpoints import build_model
from nn.optimization import _prepare_dataloaders
from utils import get_freqs
from utils.spectral_partitioning import find_peak_windows, classify_partition
from utils.spectral_peaks import find_spectral_peaks

# Config — edit and rerun.
EXPERIMENT_NAME = "shape_v13"
LEAD_TIME_HOURS = 12
SPLIT = "val"            # 'val' or 'test'
CHANNEL_SET = "full"
AUX_SET = "dmd"
BUOY_ID = "32012"
MAX_PEAKS = 4            # same cap as nn/optimization.py::objective's peak_max_count
RECALL_TOLERANCE = 2     # bins, same as utils.spectral_peaks.peak_modality_metrics
C, TAU_MIN = 0.15, 1e-4  # utils.loss.SoftPeakHeightLoss defaults

project_root = Path(__file__).resolve().parent.parent
ckpt_path = project_root / "results" / EXPERIMENT_NAME / "shape" / f"lead_{LEAD_TIME_HOURS}h" / "best_model.pt"
out_dir = project_root / "results" / "peak_window_check"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"{EXPERIMENT_NAME}_lead_{LEAD_TIME_HOURS}h_{SPLIT}.txt"

lines = []


def log(msg=""):
    print(msg)
    lines.append(msg)


ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
seq_len, lead = ckpt["params"]["seq_len"], ckpt["lead_time_steps"]

density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(
    project_root / "buoy_data" / BUOY_ID / "processed_data.pkl")
freqs = get_freqs(density)
f = freqs.numpy().astype(float)

_, val_loader, test_loader, freq_means, shape_means, *_ = _prepare_dataloaders(
    density, alpha_1, alpha_2, r_1, r_2, seq_len, lead, 256, "shape", shuffle_seed=0,
    wind=wind, channel_set=CHANNEL_SET, aux_set=AUX_SET)
loader = val_loader if SPLIT == "val" else test_loader

model = build_model(ckpt, freqs, "cpu", CHANNEL_SET, AUX_SET)
metrics, (y_pred, y_true, _) = evaluate(model, loader, "cpu", freqs, lead_time=lead,
                                        freq_means=freq_means, shape_means=shape_means,
                                        return_arrays=True, compute_peak_metrics=True)
pred = np.exp(y_pred.numpy().astype(float))  # unit-area shape, (N, lead, F)
true = np.exp(y_true.numpy().astype(float))  # floored target, as in training/evaluation

# Physical density aligned with each (sample, step) target, for the labels.
n = len(density)
train_end, val_end = int(0.7 * n), int(0.85 * n)
split_raw = (density.values[train_end:val_end] if SPLIT == "val"
             else density.values[val_end:]).astype(float)


def soft_height(E_pred, l, r, H_true):
    """SoftPeakHeightLoss's predicted height for one window, in numpy."""
    tau = max(TAU_MIN, C * H_true * (f[r] - f[l]) / (f[-1] - f[0]))
    x = E_pred[l:r + 1] / tau
    w = np.exp(x - x.max())
    return float((E_pred[l:r + 1] * w / w.sum()).sum())


def partitions(steps):
    rows = []
    for i in range(true.shape[0]):
        for s in steps:
            pp = find_spectral_peaks(f, pred[i, s])
            for p, l, r in find_peak_windows(f, true[i, s])[:MAX_PEAKS]:
                H = true[i, s, p]
                q = int(pp[np.argmin(np.abs(pp - p))]) if pp.size else None
                rows.append(dict(
                    label=classify_partition(f[p], split_raw[i + seq_len + s, p]),
                    width=r - l + 1,
                    nearest_inside=q is not None and l <= q <= r,
                    dist=abs(q - p) if q is not None else np.inf,
                    term_err=abs(soft_height(pred[i, s], l, r, H) - H) / H,
                    match_err=abs(pred[i, s, q] - H) / H if q is not None else np.nan,
                ))
    return pd.DataFrame(rows)


def report(df, title):
    log(f"\n=== {title}: {len(df)} true partitions ===")
    groups = [("all", df), ("wind_sea", df[df.label == "wind_sea"]),
              ("swell", df[df.label == "swell"]), ("width<=5 bins", df[df.width <= 5])]
    log("nearest predicted peak inside true window: "
        + "  ".join(f"{name}={g.nearest_inside.mean() * 100:.1f}% (n={len(g)})" for name, g in groups))
    bins = pd.cut(df.dist, [-1, 0, 1, RECALL_TOLERANCE, 5, np.inf],
                  labels=["0", "1", str(RECALL_TOLERANCE), f"{RECALL_TOLERANCE + 1}-5", ">5"])
    log("distance true peak -> nearest predicted peak (bins): "
        + "  ".join(f"{k}: {v * 100:.1f}%" for k, v in bins.value_counts(normalize=True, sort=False).items())
        + f"  (no predicted peak: {np.isinf(df.dist).mean() * 100:.1f}%)")
    out, ins = df[~df.nearest_inside], df[df.nearest_inside]
    log(f"median relative height error, displaced (n={len(out)}): peak term {out.term_err.median():.3f}"
        f" vs nearest predicted peak {out.match_err.median():.3f}")
    log(f"median relative height error, in window (n={len(ins)}): peak term {ins.term_err.median():.3f}")


log(f"Checkpoint: {ckpt_path.relative_to(project_root)} (trial {ckpt.get('trial_number')}, "
    f"smoothed val score {ckpt['val_score']:.4f})")
log(f"Split: {SPLIT}, lead {lead} h, seq_len {seq_len}, first {MAX_PEAKS} windows per spectrum")
log(f"Mean significant peaks per spectrum (final step): true {metrics['Peak_Count_True_Mean']:.2f}, "
    f"predicted {metrics['Peak_Count_Pred_Mean']:.2f}")
report(partitions([lead - 1]), "final step")
report(partitions(range(lead)), "all steps (what the loss sees)")

out_path.write_text("\n".join(lines) + "\n")
print(f"\nSaved {out_path}")
