"""
Visualize predicted-vs-true spectral CDFs to build intuition for the 1-D
Wasserstein (earth-mover) distance (utils.loss.SpectralWassersteinLoss) and
why it helps on multimodal (double/triple-peaked) sea states specifically.

Runs one full autoregressive test-set pass (nn.evaluate.evaluate, same as
scripts/infer.py --aggregate), then — at a single chosen forecast step —
uses utils.find_spectral_peaks to bucket every test sample by true peak
count, and picks one representative unimodal sample (peak count == 1) and
one representative multimodal sample (peak count >= 2): the one closest to
the MEDIAN per-sample Wasserstein error within its bucket, so the plot shows
a typical case rather than a cherry-picked best/worst one.

For each of the two samples, plots the PDF (E(f) or shape(f), depending on
--target) on top and the CDF on the bottom, with the area between the true
and predicted CDFs shaded — that shaded area IS the Wasserstein-1 distance
(W1 = integral of |CDF_pred - CDF_true| df, see SpectralWassersteinLoss's
docstring), so the plot makes the metric's "transport cost" reading visually
literal rather than just a number in a table.

Usage:
    python scripts/plot_cdf_wasserstein.py --experiment shape_v12 --lead 12
    python scripts/plot_cdf_wasserstein.py --experiment shape_v12 --lead 12 \
        --step -1 --save
    python scripts/plot_cdf_wasserstein.py --experiment shape_v12 --lead 12 \
        --unimodal-index 10 --multimodal-index 42
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from utils import get_freqs, get_device, find_spectral_peaks, SpectralWassersteinLoss
from nn import evaluate
from nn.checkpoints import find_checkpoint, build_model
from nn.optimization import _prepare_dataloaders
from nn.channels import CHANNEL_SETS, AUX_CHANNEL_SETS

BUOY_ID = "32012"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--experiment", required=True)
    p.add_argument("--target", default="shape", choices=["shape", "density"])
    p.add_argument("--lead", type=int, required=True, help="Lead time in hours")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--channel-set", default="full", choices=list(CHANNEL_SETS))
    p.add_argument("--aux-set", default="none", choices=list(AUX_CHANNEL_SETS))
    p.add_argument(
        "--step",
        type=int,
        default=-1,
        help="Forecast step to visualize (0-indexed, negative allowed). "
        "Default -1 = final/deliverable forecast step, matching the "
        "convention nn/evaluate.py uses for its bulk-parameter block.",
    )
    p.add_argument("--prominence-frac", type=float, default=0.25)
    p.add_argument(
        "--unimodal-index", type=int, default=None,
        help="Override auto-selected unimodal sample with this test-set index",
    )
    p.add_argument(
        "--multimodal-index", type=int, default=None,
        help="Override auto-selected multimodal sample with this test-set index",
    )
    p.add_argument("--save", action="store_true")
    return p.parse_args()


def compute_cdf(spectrum_1d, freqs_np):
    """Mass-normalized cumulative distribution, same convention as
    utils.loss.SpectralWassersteinLoss (mass-normalize, then cumulative
    trapezoid over freqs)."""
    mass = np.trapezoid(spectrum_1d, freqs_np)
    cdf = cumulative_trapezoid(spectrum_1d, freqs_np, initial=0.0)
    return cdf / max(mass, 1e-8)


def pick_representative(mask, w1_all, override_index):
    if override_index is not None:
        return override_index
    idx_pool = np.flatnonzero(mask)
    if idx_pool.size == 0:
        return None
    order = np.argsort(w1_all[idx_pool])
    median_pos = idx_pool[order[len(order) // 2]]
    return int(median_pos)


def plot_sample(ax_pdf, ax_cdf, freqs_np, true_s, pred_s, pers_s, n_peaks, w1_pred, w1_pers, ylabel):
    cdf_true = compute_cdf(true_s, freqs_np)
    cdf_pred = compute_cdf(pred_s, freqs_np)
    cdf_pers = compute_cdf(pers_s, freqs_np)

    ax_pdf.plot(freqs_np, true_s, "k-", label="True")
    ax_pdf.plot(freqs_np, pred_s, "C0-", label="Predicted")
    ax_pdf.plot(freqs_np, pers_s, "C1--", label="Persistence")
    ax_pdf.set_ylabel(ylabel)
    ax_pdf.legend(fontsize=8)
    ax_pdf.grid(True, alpha=0.3)

    ax_cdf.plot(freqs_np, cdf_true, "k-", label="True")
    ax_cdf.plot(freqs_np, cdf_pred, "C0-", label="Predicted")
    ax_cdf.plot(freqs_np, cdf_pers, "C1--", label="Persistence")
    ax_cdf.fill_between(
        freqs_np, cdf_true, cdf_pred, color="C0", alpha=0.2,
        label=f"W1(pred)={w1_pred:.4f}",
    )
    ax_cdf.set_xlabel("Frequency (Hz)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(True, alpha=0.3)

    return cdf_true, cdf_pred, cdf_pers


def main():
    args = parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    file_path = os.path.join(project_root, "buoy_data", BUOY_ID, "processed_data.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found — run scripts/data_processing.py first.")
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(file_path)
    freqs = get_freqs(density)
    freqs_np = freqs.numpy()
    device = get_device()

    ckpt, results_folder = find_checkpoint(
        Path(project_root), args.experiment, args.target, 1, args.lead, args.seed
    )
    model = build_model(ckpt, freqs, device, args.channel_set, args.aux_set)
    freq_means = ckpt["freq_means"].to(device)
    shape_means = ckpt["shape_means"].to(device) if ckpt.get("shape_means") is not None else None
    lead_time_steps = ckpt["lead_time_steps"]
    params = ckpt["params"]

    _, _, test_loader, _, _, _, _, _ = _prepare_dataloaders(
        density, alpha_1, alpha_2, r_1, r_2,
        params["seq_len"], lead_time_steps, params["batch_size"], args.target,
        shuffle_seed=args.seed, wind=wind,
        channel_set=args.channel_set, aux_set=args.aux_set,
    )

    print(f"Running full test-set evaluation ({len(test_loader.dataset)} samples)...")
    _, (y_pred_all, y_true_all, y_pers_all) = evaluate(
        model, test_loader, device, freqs, lead_time=lead_time_steps,
        freq_means=freq_means, shape_means=shape_means, return_arrays=True,
    )

    step = args.step if args.step >= 0 else lead_time_steps + args.step
    y_pred_step = y_pred_all[:, step, :]
    y_true_step = y_true_all[:, step, :]
    y_pers_step = y_pers_all[:, step, :]

    # Both 'shape' and 'density' targets are log-spectral-energy at this
    # point (see nn/evaluate.py docstring) — exp() back to physical values
    # (unit-area shape, or E(f)) before peak-finding / CDF construction.
    true_np = np.exp(y_true_step.numpy())
    pred_np = np.exp(y_pred_step.numpy())
    pers_np = np.exp(y_pers_step.numpy())

    wasserstein_fn = SpectralWassersteinLoss()
    w1_pred_all = wasserstein_fn(y_pred_step, y_true_step, freqs, reduction="none").numpy()
    w1_pers_all = wasserstein_fn(y_pers_step, y_true_step, freqs, reduction="none").numpy()

    peak_counts = np.array(
        [len(find_spectral_peaks(true_np[i], args.prominence_frac)) for i in range(true_np.shape[0])]
    )
    unimodal_mask = peak_counts == 1
    multimodal_mask = peak_counts >= 2
    print(
        f"Step {step}: {unimodal_mask.sum()} unimodal / {multimodal_mask.sum()} "
        f"multimodal / {(peak_counts == 0).sum()} no-peak samples out of {len(peak_counts)}"
    )

    uni_idx = pick_representative(unimodal_mask, w1_pred_all, args.unimodal_index)
    multi_idx = pick_representative(multimodal_mask, w1_pred_all, args.multimodal_index)
    if uni_idx is None:
        raise ValueError("No unimodal (1-peak) sample found at this step — try a different --step.")
    if multi_idx is None:
        raise ValueError("No multimodal (2+ peak) sample found at this step — try a different --step.")

    ylabel = "Shape E(f)/m₀" if args.target == "shape" else "E(f) (m²/Hz)"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    plot_sample(
        axes[0, 0], axes[1, 0], freqs_np,
        true_np[uni_idx], pred_np[uni_idx], pers_np[uni_idx],
        peak_counts[uni_idx], w1_pred_all[uni_idx], w1_pers_all[uni_idx], ylabel,
    )
    axes[0, 0].set_title(
        f"Unimodal — sample {uni_idx} ({peak_counts[uni_idx]} true peak)\n"
        f"W1(pred)={w1_pred_all[uni_idx]:.4f}  W1(persistence)={w1_pers_all[uni_idx]:.4f}"
    )

    plot_sample(
        axes[0, 1], axes[1, 1], freqs_np,
        true_np[multi_idx], pred_np[multi_idx], pers_np[multi_idx],
        peak_counts[multi_idx], w1_pred_all[multi_idx], w1_pers_all[multi_idx], ylabel,
    )
    axes[0, 1].set_title(
        f"Multimodal — sample {multi_idx} ({peak_counts[multi_idx]} true peaks)\n"
        f"W1(pred)={w1_pred_all[multi_idx]:.4f}  W1(persistence)={w1_pers_all[multi_idx]:.4f}"
    )

    fig.suptitle(
        f"{args.experiment} — {args.target} target, lead {args.lead}h, step {step + 1}\n"
        f"Shaded area between CDFs = Wasserstein-1 distance (earth-mover transport cost)"
    )
    fig.tight_layout()

    if args.save:
        save_dir = results_folder / "inference_plots"
        save_dir.mkdir(exist_ok=True)
        out_path = save_dir / f"cdf_wasserstein_step{step}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot → {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
