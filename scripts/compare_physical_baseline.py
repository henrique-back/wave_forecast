"""
Physical-model baseline at NDBC 32012: the GEFSv12 wave reforecast (control
member, WAVEWATCH III) scored against the transformer 'shape' model, the
ridge-AR baseline and persistence, on identical start times and identical
truth, with the same final-step shape panel
(nn/spectrum_eval.py::compute_shape_final_metrics).

Method, fixed before any GEFS score was looked at:
  * Start times are the GEFS 00Z cycle times T inside the test split, kept
    only if every compared model has a full input window before T (the
    largest of the transformer's seq_len and the AR order) and T + lead is
    still inside the test split. Each model's own forecast issued at T is
    taken at its final step (valid time T + lead).
  * Band: the NDBC bins covered by the GEFS grid (0.0375-0.485 Hz, 45 bins;
    the 0.02 and 0.0325 Hz bins are dropped). GEFS is interpolated linearly,
    in linear E, onto those bins. Every spectrum (truth, persistence, each
    forecast) is clipped at 0, restricted to the band and renormalised to unit
    area on it (the ridge AR can predict negative bins; the count is reported).
  * Truth is the raw buoy density at T + lead and persistence the raw density
    at T, without nn/evaluate.py's 1e-3*shape_means floor, so the transformer's
    numbers here differ slightly from evaluate()'s own.
  * Wind-sea/swell labels are computed on the physical buoy density (gamma*
    is defined in m^2/Hz), as nn/evaluate.py now also does; see
    manuscript/decisions/log/029. The transformer is also scored with the old
    unit-area-shape labels, as context.
  * GEFS has no lead-0 output; its earliest lead (+3 h) is scored as the
    model-vs-buoy mismatch at initialisation. It is not an analysis: the
    reforecast assimilates no wave data.
  * Uncertainty: paired circular block bootstrap over start dates (blocks of
    BLOCK_DAYS consecutive cycles), on each model's difference from GEFS.

Inputs: buoy_data/32012/processed_data.pkl, buoy_data/32012/gefsv12_c00_spec1d.npz
(scripts/fetch_gefs_reforecast.py), results/<experiment>/shape/lead_<N>h/best_model.pt,
results/linear_baseline/shape/lead_<N>h/linear_baseline_final.pt.
Output: results/comparisons/physical_baseline_gefsv12/lead_<N>h/{metrics.json, summary.md, arrays.npz}.

Usage:
    python scripts/compare_physical_baseline.py --experiment shape_v13 --leads 12 24 48
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from nn import evaluate
from nn.checkpoints import build_model
from nn.optimization import _prepare_dataloaders, _compute_val_score
from nn.spectrum_eval import compute_shape_final_metrics
from utils import get_freqs, compute_shape
from utils.linear_baseline import forecast_coeffs
from utils.spectral_peaks import find_spectral_peaks

BUOY_ID = "32012"
CHANNEL_SET = "full"
AUX_SET = "dmd"
GEFS_NAME = "GEFSv12_c00"
BLOCK_DAYS = 3

# Bootstrapped metrics and whether lower is better.
BOOT_METRICS = {
    "Shape_RMSE": True,
    "Shape_Wasserstein": True,
    "Tm02_RMSE": True,
    "peak_fidelity": False,
    "Peak_Separation_Recall_windsea": False,
    "Peak_Separation_Recall_swell": False,
    "Peak_Height_RelError_windsea": True,
    "Peak_Height_RelError_swell": True,
}
SUMMARY_METRICS = ["Shape_RMSE", "Shape_SS", "Shape_Wasserstein", "Tm02_RMSE", "Tm02_Bias",
                   "peak_fidelity", "Peak_Separation_Recall_windsea",
                   "Peak_Separation_Recall_swell", "Peak_Height_RelError_windsea",
                   "Peak_Height_RelError_swell", "Peak_Count_True_Mean", "Peak_Count_Pred_Mean"]

project_root = Path(__file__).resolve().parent.parent


def band_mask(buoy_freqs, model_freqs):
    """Buoy bins inside the model's frequency range."""
    band = (buoy_freqs >= model_freqs.min()) & (buoy_freqs <= model_freqs.max())
    if band.sum() < 2:
        raise ValueError("fewer than two buoy bins inside the model frequency range")
    return band


def band_shape(spectra, freqs, band):
    """Clip at 0, keep the band bins, renormalise to unit area on the band."""
    return compute_shape(np.clip(spectra[..., band], 0.0, None), freqs[band])


def interp_to(spectra, src_freqs, dst_freqs):
    """Linear interpolation of each row onto dst_freqs, which must lie inside src_freqs."""
    if dst_freqs.min() < src_freqs.min() or dst_freqs.max() > src_freqs.max():
        raise ValueError("target frequencies fall outside the source grid")
    return np.stack([np.interp(dst_freqs, src_freqs, row) for row in spectra])


def eligible_rows(index, init_times, val_end, window, lead):
    """Buoy row k of each start time and whether every model can forecast from it."""
    k = index.get_indexer(init_times)
    ok = (k >= 0) & (k - window + 1 >= val_end) & (k + lead <= len(index) - 1)
    return k, ok


def block_bootstrap_indices(n, block, n_boot, rng):
    """Circular block bootstrap: (n_boot, n) resampled sample indices."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return idx.reshape(n_boot, -1)[:, :n]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _fmt(x):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"


def run_lead(lead, experiment, n_boot, data, gefs):
    density, alpha_1, alpha_2, r_1, r_2, wind = data
    index = density.index
    raw = density.to_numpy(dtype=np.float64)
    buoy_f = np.array([float(c) for c in density.columns])
    freqs_t = get_freqs(density)
    if not np.allclose(buoy_f, freqs_t.numpy()):
        raise ValueError("pickle column order does not match get_freqs")
    val_end = int(0.85 * len(density))          # nn/optimization.py::_prepare_dataloaders

    ckpt_path = project_root / "results" / experiment / "shape" / f"lead_{lead}h" / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ar_path = project_root / "results" / "linear_baseline" / "shape" / f"lead_{lead}h" / "linear_baseline_final.pt"
    ar = torch.load(ar_path, map_location="cpu", weights_only=False)
    if ckpt["lead_time_steps"] != lead or ar["lead_time_steps"] != lead:
        raise ValueError(f"checkpoint lead mismatch for {lead} h")
    seq_len, order = ckpt["params"]["seq_len"], ar["order"]

    gefs_f = gefs["freqs"].astype(np.float64)
    (li,) = np.flatnonzero(gefs["lead_hours"] == lead)
    (li3,) = np.flatnonzero(gefs["lead_hours"] == 3)
    k, ok = eligible_rows(index, pd.DatetimeIndex(gefs["init_times"]), val_end, max(seq_len, order), lead)
    ok &= np.isfinite(gefs["E1d"][:, li]).all(axis=1)
    cyc, k = np.flatnonzero(ok), k[ok]
    issue_times = index[k]
    n = len(k)

    band = band_mask(buoy_f, gefs_f)
    fb = buoy_f[band]
    truth = band_shape(raw[k + lead], buoy_f, band)
    pers = band_shape(raw[k], buoy_f, band)
    m0_true = np.trapezoid(np.clip(raw[k + lead][:, band], 0.0, None), fb, axis=1)

    # Transformer: its own test windows, subset to the start rows.
    _, _, test_loader, freq_means, shape_means, *_ = _prepare_dataloaders(
        density, alpha_1, alpha_2, r_1, r_2, seq_len, lead, 256, "shape", shuffle_seed=0,
        wind=wind, channel_set=CHANNEL_SET, aux_set=AUX_SET)
    if not (torch.allclose(freq_means, ckpt["freq_means"]) and torch.allclose(shape_means, ckpt["shape_means"])):
        raise ValueError("loader normalisation differs from the checkpoint's")
    j = k - val_end - seq_len + 1
    subset = Subset(test_loader.dataset, j.tolist())
    subset.m0_true = test_loader.dataset.m0_true[j]  # Subset hides it; evaluate() labels need it
    loader = DataLoader(subset, batch_size=256, shuffle=False)
    model = build_model(ckpt, freqs_t, "cpu", CHANNEL_SET, AUX_SET)
    eval_metrics, (y_pred, y_true, _) = evaluate(
        model, loader, "cpu", freqs_t, lead_time=lead, freq_means=freq_means,
        shape_means=shape_means, return_arrays=True, compute_peak_metrics=True)
    tr_pred = np.exp(y_pred[:, -1].numpy().astype(np.float64))
    expected = np.maximum(compute_shape(raw[k + lead], buoy_f), 1e-3 * shape_means.numpy())
    if not np.allclose(np.exp(y_true[:, -1].numpy().astype(np.float64)), expected, rtol=1e-4, atol=1e-7):
        raise AssertionError("transformer targets are not the buoy spectra at T + lead (alignment)")

    # Ridge AR: its own test windows, selected by start row.
    ar_pred_all, ar_true_all, _ = forecast_coeffs(density, ar["freqs"], ar["coeffs"], order, lead,
                                                  "shape", eval_split="test")
    i_ar = k - val_end - order + 1
    if not np.allclose(ar_true_all[i_ar, -1], compute_shape(raw[k + lead], buoy_f), rtol=1e-4, atol=1e-9):
        raise AssertionError("AR targets are not the buoy spectra at T + lead (alignment)")
    ar_pred = ar_pred_all[i_ar, -1]

    gefs_e = interp_to(gefs["E1d"][cyc, li].astype(np.float64), gefs_f, fb)
    forecasts = {
        GEFS_NAME: compute_shape(np.clip(gefs_e, 0.0, None), fb),
        experiment: band_shape(tr_pred, buoy_f, band),
        "ridge_AR": band_shape(ar_pred, buoy_f, band),
        "persistence": pers,
    }
    metrics = {name: compute_shape_final_metrics(fb, fc, truth, pers, m0_true=m0_true)
               for name, fc in forecasts.items()}

    # Context rows.
    old_labels = compute_shape_final_metrics(fb, forecasts[experiment], truth, pers, m0_true=None)
    gefs3 = compute_shape(np.clip(interp_to(gefs["E1d"][cyc, li3].astype(np.float64), gefs_f, fb), 0.0, None), fb)
    lead3 = compute_shape_final_metrics(fb, gefs3, band_shape(raw[k + 3], buoy_f, band), pers,
                                        m0_true=np.trapezoid(np.clip(raw[k + 3][:, band], 0.0, None), fb, axis=1))
    hs_true = 4 * np.sqrt(m0_true)
    hs_gefs = 4 * np.sqrt(np.trapezoid(np.clip(gefs_e, 0.0, None), fb, axis=1))
    hs_pers = 4 * np.sqrt(np.trapezoid(np.clip(raw[k][:, band], 0.0, None), fb, axis=1))
    n_peaks_full = sum(len(find_spectral_peaks(buoy_f, s)) for s in raw[k + lead])
    n_peaks_band = sum(len(find_spectral_peaks(fb, s)) for s in np.clip(raw[k + lead][:, band], 0, None))
    context = {
        "transformer_old_labels": {m: old_labels[m] for m in BOOT_METRICS if m in old_labels},
        "transformer_evaluate_own": {
            "Shape_RMSE": eval_metrics["Shape_RMSE"], "Tm02_RMSE": eval_metrics["Tm02_RMSE"],
            "Shape_Wasserstein": eval_metrics["Shape_Wasserstein"],
            "peak_fidelity": _compute_val_score(eval_metrics, "peak_fidelity"),
            "note": "evaluate() on the same start times: full 47-bin grid, floored truth, physical labels",
        },
        "GEFS_lead3h": {m: lead3[m] for m in SUMMARY_METRICS if m in lead3},
        "Hs_band": {
            "GEFS_RMSE": float(np.sqrt(np.mean((hs_gefs - hs_true) ** 2))),
            "GEFS_Bias": float(np.mean(hs_gefs - hs_true)),
            "persistence_RMSE": float(np.sqrt(np.mean((hs_pers - hs_true) ** 2))),
            "persistence_Bias": float(np.mean(hs_pers - hs_true)),
        },
        "ridge_AR_samples_with_negative_bins": int((ar_pred < 0).any(axis=1).sum()),
        "true_peaks_full_grid": int(n_peaks_full),
        "true_peaks_band": int(n_peaks_band),
    }

    # Paired circular block bootstrap of each model's difference from GEFS.
    rng = np.random.default_rng(0)
    idx = block_bootstrap_indices(n, BLOCK_DAYS, n_boot, rng)
    draws = {name: {m: np.empty(n_boot) for m in BOOT_METRICS} for name in forecasts}
    for b, ib in enumerate(idx):
        for name, fc in forecasts.items():
            mb = compute_shape_final_metrics(fb, fc[ib], truth[ib], pers[ib], m0_true=m0_true[ib])
            for m in BOOT_METRICS:
                draws[name][m][b] = mb[m]
    bootstrap = {}
    for name in forecasts:
        if name == GEFS_NAME:
            continue
        bootstrap[name] = {}
        for m, lower_better in BOOT_METRICS.items():
            d = draws[name][m] - draws[GEFS_NAME][m]
            better = d < 0 if lower_better else d > 0
            bootstrap[name][m] = {
                "diff": metrics[name][m] - metrics[GEFS_NAME][m],
                "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                "p_better": float(np.mean(better[np.isfinite(d)])) if np.isfinite(d).any() else None,
            }

    out_dir = project_root / "results" / "comparisons" / "physical_baseline_gefsv12" / f"lead_{lead}h"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "lead_h": lead, "n_samples": n, "issue_times": [str(t) for t in issue_times],
        "band_hz": [float(fb[0]), float(fb[-1]), int(band.sum())],
        "checkpoints": {experiment: str(ckpt_path.relative_to(project_root)),
                        "ridge_AR": str(ar_path.relative_to(project_root))},
        "seq_len": seq_len, "ar_order": order, "n_boot": n_boot, "block_days": BLOCK_DAYS,
    }
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(_jsonable({"meta": meta, "models": metrics, "context": context,
                             "bootstrap_vs_GEFS": bootstrap}), fh, indent=2)
    np.savez_compressed(out_dir / "arrays.npz", freqs=fb, issue_times=issue_times.to_numpy(),
                        truth=truth, m0_true=m0_true, **forecasts)

    lines = [f"# Physical baseline at {BUOY_ID}, lead {lead} h", "",
             f"N = {n} start times ({issue_times[0]:%Y-%m-%d} to {issue_times[-1]:%Y-%m-%d}, 00Z); "
             f"band {fb[0]:.4f}-{fb[-1]:.4f} Hz ({band.sum()} bins); labels on the physical density.", "",
             "| Model | " + " | ".join(SUMMARY_METRICS) + " |",
             "|" + "---|" * (len(SUMMARY_METRICS) + 1)]
    for name, m in metrics.items():
        lines.append(f"| {name} | " + " | ".join(_fmt(m.get(k)) for k in SUMMARY_METRICS) + " |")
    lines += ["", f"Difference from {GEFS_NAME} (model − GEFS), 95% block-bootstrap interval, "
              f"P(model better); B = {n_boot}, blocks of {BLOCK_DAYS} days.", "",
              "| Model | Metric | Diff | 95% CI | P(better) |", "|---|---|---|---|---|"]
    for name, rows in bootstrap.items():
        for m, r in rows.items():
            lines.append(f"| {name} | {m} | {_fmt(r['diff'])} | [{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}] "
                         f"| {_fmt(r['p_better'])} |")
    lines += ["", "Context:", "",
              f"- GEFS at +3 h (initialisation mismatch, not an analysis): Shape_RMSE "
              f"{_fmt(lead3['Shape_RMSE'])}, Tm02_RMSE {_fmt(lead3['Tm02_RMSE'])}, PF {_fmt(lead3['peak_fidelity'])}",
              f"- Band Hs: GEFS RMSE {_fmt(context['Hs_band']['GEFS_RMSE'])} m, bias "
              f"{_fmt(context['Hs_band']['GEFS_Bias'])} m; persistence RMSE {_fmt(context['Hs_band']['persistence_RMSE'])} m",
              f"- {experiment} with the old unit-area-shape labels: PF {_fmt(old_labels['peak_fidelity'])}",
              f"- Ridge AR forecasts with negative bins (clipped): {context['ridge_AR_samples_with_negative_bins']} of {n}",
              f"- True significant peaks: {n_peaks_full} on the full grid, {n_peaks_band} on the band",
              f"- Checkpoints: {meta['checkpoints']}"]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"lead {lead} h: N = {n}, written to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", default="shape_v13")
    parser.add_argument("--leads", type=int, nargs="+", default=[12, 24, 48])
    parser.add_argument("--n-boot", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_pickle(project_root / "buoy_data" / BUOY_ID / "processed_data.pkl")
    gefs = np.load(project_root / "buoy_data" / BUOY_ID / "gefsv12_c00_spec1d.npz")
    for lead in args.leads:
        run_lead(lead, args.experiment, args.n_boot, data, gefs)


if __name__ == "__main__":
    main()
