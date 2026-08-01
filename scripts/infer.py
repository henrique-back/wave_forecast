"""
Inspect a trained model's predictions on the held-out test set.

Loads checkpoint(s) saved by scripts/train.py (final_model_seed{N}.pt) — or,
if that hasn't been run yet, falls back to the Optuna search's own
best_model.pt — runs autoregressive inference on selected test-set samples,
prints per-sample bulk parameters (Hs, Tm02) across the forecast horizon, and
plots predicted vs true vs persistence.

--target combined loads a separate 'hs' checkpoint and 'shape' checkpoint
(same experiment/lead) and recombines them into a physical spectrum:
    E_pred(f, t) = shape_pred(f, t) * m0_pred(t),  m0_pred = (Hs_pred / 4)^2
This is the shape/magnitude model split described in CLAUDE.md — see the
project discussion of why a single density-target model tends to
underestimate spectral peaks and over-smooth the high-frequency tail.

--save-metrics additionally runs a full pass over the ENTIRE test set (not
just the inspected samples) and writes the resulting metrics into
results/inference_metrics/{experiment}.json, nested under [target][lead_Nh]
— the same metric set scripts/compare_versions.py computes, but for a single
experiment, so a self-comparison run is no longer needed just to get a
study's own numbers on file.

--target shape --aggregate runs a full test-set pass and plots mean +/- std
of S_true(f)/S_pred(f)/S_persistence(f) for one forecast step across every
test sample, plus prints Shape_RMSE/Shape_SS/Shape_Mass_Error — the aggregate
counterpart to the per-sample --target shape plots.

Usage:
    python scripts/infer.py --experiment weightedmeanSS_conv_freqemb_v3 --lead 6
    python scripts/infer.py --experiment weightedmeanSS_conv_freqemb_v3 --lead 6 \
        --index 42 --save
    python scripts/infer.py --experiment hs_shape_v5 --target combined --lead 6 \
        --save-metrics
    python scripts/infer.py --experiment shape_v8 --target shape --lead 6 \
        --aggregate
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
import matplotlib.pyplot as plt

from utils import (
    get_freqs,
    compute_bulk_params,
    get_start_token,
    get_device,
    trapz_weights,
)
from nn import evaluate
from nn.checkpoints import find_checkpoint, build_model
from nn.spectrum_eval import eval_combined, compute_density_metrics
from nn.optimization import _prepare_dataloaders
from nn.channels import CHANNEL_SETS, AUX_CHANNEL_SETS

BUOY_ID = "32012"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--experiment",
        required=True,
        help="EXPERIMENT_NAME the checkpoint was trained under (results/{experiment}/...)",
    )
    p.add_argument(
        "--target",
        default="density",
        choices=["density", "hs", "shape", "combined"],
        help="'combined' loads both an 'hs' and a 'shape' checkpoint under the same "
        "experiment/lead and recombines them into a physical spectrum",
    )
    p.add_argument("--lead", type=int, required=True, help="Lead time in hours")
    p.add_argument(
        "--seed", type=int, default=42, help="Seed suffix of the checkpoint to load"
    )
    p.add_argument(
        "--channel-set",
        default="full",
        choices=list(CHANNEL_SETS),
        help="Must match the CHANNEL_SET the checkpoint(s) were trained with — "
        "not stored in the checkpoint itself, see scripts/train.py",
    )
    p.add_argument(
        "--aux-set",
        default="none",
        choices=list(AUX_CHANNEL_SETS),
        help="Must match the AUX_SET the checkpoint(s) were trained with",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Specific sample index to inspect (overrides --n-samples/--random). "
        "For --target combined this indexes the common valid forecast-start "
        "range shared by both checkpoints, not either one's raw dataset index.",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=3,
        help="Number of samples to inspect when --index is not given",
    )
    p.add_argument(
        "--random",
        action="store_true",
        help="Pick random sample indices instead of evenly spaced across the test set",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=4,
        help="Number of forecast-horizon spectrum snapshots to plot per sample "
        "(density/shape/combined targets only)",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Also save each figure as PNG under the checkpoint folder",
    )
    p.add_argument(
        "--save-metrics",
        action="store_true",
        help="Also run a full test-set evaluation (all samples, not just the "
        "inspected ones) and write/update results/inference_metrics/"
        "{experiment}.json with this target/lead's metrics",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="--target shape only: instead of per-sample plots, run a full "
        "test-set pass and plot mean +/- std of S_true/S_pred/S_persistence "
        "across every test sample for one forecast step (see --agg-step), "
        "plus print Shape_RMSE/Shape_SS/Shape_Mass_Error.",
    )
    p.add_argument(
        "--agg-step",
        type=int,
        default=0,
        help="Forecast step (0-indexed) to summarize when --aggregate is set",
    )
    return p.parse_args()


def save_metrics_json(
    project_root, experiment, target, lead, seed, channel_set, aux_set, metrics
):
    out_path = project_root / "results" / "inference_metrics" / f"{experiment}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out_path.read_text()) if out_path.exists() else {}
    data.setdefault(target, {})[f"lead_{lead}h"] = {
        "seed": seed,
        "channel_set": channel_set,
        "aux_set": aux_set,
        "metrics": metrics,
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Saved full-test-set metrics → {out_path} [{target}/lead_{lead}h]")


def select_indices(n_total, n_samples, index, random_pick, seed):
    if index is not None:
        return [index]
    n_samples = min(n_samples, n_total)
    if random_pick:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(n_total, size=n_samples, replace=False).tolist())
    return sorted(set(np.linspace(0, n_total - 1, n_samples, dtype=int).tolist()))


def plot_density_sample(
    idx,
    freqs_np,
    steps,
    pred_phys,
    true_phys,
    pers_phys,
    hs_pred,
    hs_true,
    hs_pers,
    title_prefix="Sample",
):
    lead_time = pred_phys.shape[0]
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(4 * (len(steps) + 1), 4))

    ax = axes[0]
    x = np.arange(1, lead_time + 1)
    ax.plot(x, hs_true, "k-o", label="True", markersize=4)
    ax.plot(x, hs_pred, "C0-o", label="Predicted", markersize=4)
    ax.plot(x, hs_pers, "C1--", label="Persistence", markersize=4)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Hs (m)")
    ax.set_title(f"{title_prefix} {idx} — Hs over horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax, step in zip(axes[1:], steps):
        ax.plot(freqs_np, true_phys[step], "k-", label="True")
        ax.plot(freqs_np, pred_phys[step], "C0-", label="Predicted")
        ax.plot(freqs_np, pers_phys[step], "C1--", label="Persistence")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("E(f) (m²/Hz)")
        ax.set_title(
            f"Step {step + 1} (Hs true={hs_true[step]:.2f} pred={hs_pred[step]:.2f})"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_shape_sample(idx, freqs_np, steps, pred_shape, true_shape, pers_shape):
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 4))
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        ax.plot(freqs_np, true_shape[step], "k-", label="True")
        ax.plot(freqs_np, pred_shape[step], "C0-", label="Predicted")
        ax.plot(freqs_np, pers_shape[step], "C1--", label="Persistence")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Shape E(f)/m₀ (unit area)")
        ax.set_title(f"Sample {idx} — step {step + 1}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_shape_test_set_summary(freqs_np, y_pred_all, y_true_all, y_pers_all, step):
    """Mean +/- std of S_true/S_pred/S_persistence across the whole test set,
    for one forecast step view.

    y_pred_all/y_true_all/y_pers_all: (total_samples, lead_time, num_freqs)
    numpy arrays, as returned by nn.evaluate.evaluate(..., return_arrays=True).
    """
    pred = y_pred_all[:, step, :]
    true = y_true_all[:, step, :]
    pers = y_pers_all[:, step, :]

    fig, ax = plt.subplots(figsize=(7, 5))
    for arr, color, label in [
        (true, "k", "True"),
        (pred, "C0", "Predicted"),
        (pers, "C1", "Persistence"),
    ]:
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(freqs_np, mean, color=color, label=label)
        ax.fill_between(freqs_np, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Shape E(f)/m₀ (unit area)")
    ax.set_title(
        f"Test-set S_true vs S_pred — step {step + 1} (mean ± std, n={pred.shape[0]})"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_hs_sample(idx, hs_pred, hs_true, hs_pers):
    lead_time = len(hs_true)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, lead_time + 1)
    ax.plot(x, hs_true, "k-o", label="True", markersize=4)
    ax.plot(x, hs_pred, "C0-o", label="Predicted", markersize=4)
    ax.plot(x, hs_pers, "C1--", label="Persistence", markersize=4)
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Hs (m)")
    ax.set_title(f"Sample {idx} — Hs over horizon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def print_bulk_table(
    idx, lead_time_steps, hs_true, hs_pred, hs_pers, tm02_true=None, tm02_pred=None
):
    print(f"\nSample {idx}:")
    if tm02_true is not None:
        print(
            f"  {'step':>4} {'Hs true':>8} {'Hs pred':>8} {'Hs pers':>8} "
            f"{'Tm02 true':>10} {'Tm02 pred':>10}"
        )
        for s in range(lead_time_steps):
            print(
                f"  {s + 1:>4} {hs_true[s]:>8.3f} {hs_pred[s]:>8.3f} {hs_pers[s]:>8.3f} "
                f"{tm02_true[s]:>10.3f} {tm02_pred[s]:>10.3f}"
            )
    else:
        print(f"  {'step':>4} {'Hs true':>8} {'Hs pred':>8} {'Hs pers':>8}")
        for s in range(lead_time_steps):
            print(
                f"  {s + 1:>4} {hs_true[s]:>8.3f} {hs_pred[s]:>8.3f} {hs_pers[s]:>8.3f}"
            )


def run_single(
    args, project_root, density, alpha_1, alpha_2, r_1, r_2, wind, freqs, freqs_np, device
):
    ckpt, results_folder = find_checkpoint(
        project_root, args.experiment, args.target, 1, args.lead, args.seed
    )
    model = build_model(ckpt, freqs, device, args.channel_set, args.aux_set)
    freq_means = ckpt["freq_means"].to(device)
    shape_means = ckpt["shape_means"].to(device) if ckpt.get("shape_means") is not None else None
    lead_time_steps = ckpt["lead_time_steps"]
    params = ckpt["params"]

    # shuffle_seed only affects the (unused) train_loader's shuffle order —
    # the test split itself is deterministic and never shuffled.
    _, _, test_loader, _, _, _, _, _ = _prepare_dataloaders(
        density,
        alpha_1,
        alpha_2,
        r_1,
        r_2,
        params["seq_len"],
        lead_time_steps,
        params["batch_size"],
        args.target,
        shuffle_seed=args.seed,
        wind=wind,
        channel_set=args.channel_set,
        aux_set=args.aux_set,
    )
    test_dataset = test_loader.dataset

    n_total = len(test_dataset)

    if args.aggregate:
        print(
            f"Running full test-set evaluation for aggregate shape summary "
            f"({n_total} samples)..."
        )
        agg_metrics, (y_pred_all, y_true_all, y_pers_all) = evaluate(
            model,
            test_loader,
            device,
            freqs,
            lead_time=lead_time_steps,
            freq_means=freq_means,
            shape_means=shape_means,
            return_arrays=True,
        )
        print(
            f"Shape_RMSE={agg_metrics['Shape_RMSE']:.4f}  "
            f"Shape_SS={agg_metrics['Shape_SS']:.4f}  "
            f"Shape_Mass_Error={agg_metrics['Shape_Mass_Error']:.6f}"
        )
        # evaluate()'s returned arrays are log-shape for target=='shape'
        # (see its docstring NOTE) — exp() back to linear before plotting.
        fig = plot_shape_test_set_summary(
            freqs_np,
            np.exp(y_pred_all.numpy()),
            np.exp(y_true_all.numpy()),
            np.exp(y_pers_all.numpy()),
            args.agg_step,
        )
        if args.save:
            save_dir = results_folder / "inference_plots"
            save_dir.mkdir(exist_ok=True)
            out_path = save_dir / f"shape_test_set_summary_step{args.agg_step}.png"
            fig.savefig(out_path, dpi=150)
            print(f"  Saved plot → {out_path}")
        plt.show()
        return

    indices = select_indices(
        n_total, args.n_samples, args.index, args.random, args.seed
    )
    print(f"Inspecting {len(indices)} test sample(s) out of {n_total}: {indices}")

    if args.save_metrics:
        print(f"Running full test-set evaluation ({n_total} samples)...")
        full_metrics = evaluate(
            model,
            test_loader,
            device,
            freqs,
            lead_time=lead_time_steps,
            freq_means=freq_means,
            shape_means=shape_means,
            compute_peak_metrics=(args.target == "shape"),
        )
        save_metrics_json(
            project_root,
            args.experiment,
            args.target,
            args.lead,
            args.seed,
            args.channel_set,
            args.aux_set,
            full_metrics,
        )

    save_dir = None
    if args.save:
        save_dir = results_folder / "inference_plots"
        save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for idx in indices:
            X, aux, y_true = test_dataset[idx]
            X = X.unsqueeze(0).to(device)
            aux = aux.unsqueeze(0).to(device)
            y_true = y_true.unsqueeze(0).to(device)

            y_pred = model.infer(
                X, freqs, lead_time_steps, freq_means=freq_means,
                shape_means=shape_means, aux=aux
            )
            start_token = get_start_token(
                X, args.target, freqs, device, freq_means=freq_means,
                shape_means=shape_means
            )
            persistence = start_token.unsqueeze(1).expand(-1, y_true.shape[1], -1)

            if args.target == "density":
                fm = freq_means.cpu().numpy()
                # y_pred/persistence are log-spectral-energy; y_true is the
                # raw dataset ground truth (still Ẽ = E/μ(f), unaffected).
                pred_phys = np.exp(y_pred.cpu().numpy()[0])
                true_phys = y_true.cpu().numpy()[0] * fm
                pers_phys = np.exp(persistence.cpu().numpy()[0])

                hs_pred, tm02_pred = compute_bulk_params(
                    pred_phys[np.newaxis], freqs_np
                )
                hs_true, tm02_true = compute_bulk_params(
                    true_phys[np.newaxis], freqs_np
                )
                hs_pers, tm02_pers = compute_bulk_params(
                    pers_phys[np.newaxis], freqs_np
                )
                hs_pred, hs_true, hs_pers = hs_pred[0], hs_true[0], hs_pers[0]
                tm02_pred, tm02_true = tm02_pred[0], tm02_true[0]

                print_bulk_table(
                    idx,
                    lead_time_steps,
                    hs_true,
                    hs_pred,
                    hs_pers,
                    tm02_true,
                    tm02_pred,
                )

                n_snap = min(args.n_steps, lead_time_steps)
                steps = sorted(
                    set(np.linspace(0, lead_time_steps - 1, n_snap, dtype=int).tolist())
                )
                fig = plot_density_sample(
                    idx,
                    freqs_np,
                    steps,
                    pred_phys,
                    true_phys,
                    pers_phys,
                    hs_pred,
                    hs_true,
                    hs_pers,
                )

            elif args.target == "shape":
                # y_pred/persistence are log-shape; y_true is the raw
                # dataset ground truth (already physical unit-area shape,
                # per prepare_y — unaffected by the ablation).
                pred_shape = np.exp(y_pred.cpu().numpy()[0])
                true_shape = y_true.cpu().numpy()[0]
                pers_shape = np.exp(persistence.cpu().numpy()[0])

                n_snap = min(args.n_steps, lead_time_steps)
                steps = sorted(
                    set(np.linspace(0, lead_time_steps - 1, n_snap, dtype=int).tolist())
                )
                fig = plot_shape_sample(
                    idx, freqs_np, steps, pred_shape, true_shape, pers_shape
                )
                # Frequency-weighted (trapezoidal), matching the training
                # loss/nn.evaluate.py convention since the v7->v8 bump —
                # not a flat mean over the log-spaced frequency grid.
                freq_w = trapz_weights(freqs_np)
                shape_rmse_per_step = np.sqrt(
                    (((pred_shape - true_shape) ** 2) * freq_w).sum(axis=1)
                )
                print(
                    f"\nSample {idx}: shape RMSE per step (freq-weighted) = "
                    f"{shape_rmse_per_step}"
                )

            else:  # hs
                hs_pred = y_pred.cpu().numpy()[0, :, 0]
                hs_true = y_true.cpu().numpy()[0, :, 0]
                hs_pers = persistence.cpu().numpy()[0, :, 0]
                print_bulk_table(idx, lead_time_steps, hs_true, hs_pred, hs_pers)
                fig = plot_hs_sample(idx, hs_pred, hs_true, hs_pers)

            if save_dir is not None:
                out_path = save_dir / f"sample_{idx}.png"
                fig.savefig(out_path, dpi=150)
                print(f"  Saved plot → {out_path}")

    plt.show()


def run_combined(
    args, project_root, density, alpha_1, alpha_2, r_1, r_2, wind, freqs, freqs_np, device
):
    hs_ckpt, hs_folder = find_checkpoint(
        project_root, args.experiment, "hs", 1, args.lead, args.seed
    )
    shape_ckpt, shape_folder = find_checkpoint(
        project_root, args.experiment, "shape", 1, args.lead, args.seed
    )

    lead_time_steps = hs_ckpt["lead_time_steps"]
    if shape_ckpt["lead_time_steps"] != lead_time_steps:
        raise ValueError(
            f"hs checkpoint lead_time_steps={lead_time_steps} != "
            f"shape checkpoint lead_time_steps={shape_ckpt['lead_time_steps']}"
        )

    hs_model = build_model(hs_ckpt, freqs, device, args.channel_set, args.aux_set)
    shape_model = build_model(shape_ckpt, freqs, device, args.channel_set, args.aux_set)
    hs_freq_means = hs_ckpt["freq_means"].to(device)
    shape_freq_means = shape_ckpt["freq_means"].to(device)
    shape_means = shape_ckpt["shape_means"].to(device)
    hs_params = hs_ckpt["params"]
    shape_params = shape_ckpt["params"]

    # Each model gets its own DataLoader built with its OWN seq_len — seq_len
    # is an independently tuned hyperparameter per target (here hs=48, shape=12
    # for the 6h study), so the two test datasets are windowed differently and
    # dataset index i does NOT refer to the same forecast time in both.
    _, _, hs_test_loader, _, _, _, _, _ = _prepare_dataloaders(
        density,
        alpha_1,
        alpha_2,
        r_1,
        r_2,
        hs_params["seq_len"],
        lead_time_steps,
        hs_params["batch_size"],
        "hs",
        shuffle_seed=args.seed,
        wind=wind,
        channel_set=args.channel_set,
        aux_set=args.aux_set,
    )
    _, _, shape_test_loader, _, _, _, _, _ = _prepare_dataloaders(
        density,
        alpha_1,
        alpha_2,
        r_1,
        r_2,
        shape_params["seq_len"],
        lead_time_steps,
        shape_params["batch_size"],
        "shape",
        shuffle_seed=args.seed,
        wind=wind,
        channel_set=args.channel_set,
        aux_set=args.aux_set,
    )
    hs_dataset = hs_test_loader.dataset
    shape_dataset = shape_test_loader.dataset

    # Ground truth / persistence come directly from the raw physical test
    # split — same deterministic 70/15/15 positional split _prepare_dataloaders
    # uses internally — rather than either model's own (differently windowed)
    # y, so both are unambiguous regardless of seq_len.
    n = len(density)
    val_end = int(0.85 * n)
    test_density_phys = density[val_end:]

    # t0 = absolute row index (within the test split) of the first forecast
    # step. Sample i of a dataset built with seq_len S has forecast start
    # t0 = i + S, so the two models' indices that share the same t0 are
    # idx_hs = t0 - seq_len_hs and idx_shape = t0 - seq_len_shape.
    t0_min = max(hs_params["seq_len"], shape_params["seq_len"])
    t0_max = len(test_density_phys) - lead_time_steps - 1
    n_valid = t0_max - t0_min + 1
    if n_valid <= 0:
        raise ValueError(
            "No overlapping forecast-start range between the hs and shape test splits."
        )

    rel_indices = select_indices(
        n_valid, args.n_samples, args.index, args.random, args.seed
    )
    t0_values = [t0_min + r for r in rel_indices]
    print(
        f"Inspecting {len(t0_values)} combined sample(s) out of {n_valid} valid: {t0_values}"
    )

    if args.save_metrics:
        print(f"Running full test-set evaluation ({n_valid} valid forecast starts)...")
        pred_np, true_np, pers_np, _, _ = eval_combined(
            project_root,
            args.experiment,
            1,
            args.lead,
            args.seed,
            density,
            alpha_1,
            alpha_2,
            r_1,
            r_2,
            wind,
            freqs,
            device,
            args.channel_set,
            args.aux_set,
        )
        full_metrics = compute_density_metrics(pred_np, true_np, pers_np, freqs_np)
        save_metrics_json(
            project_root,
            args.experiment,
            "combined",
            args.lead,
            args.seed,
            args.channel_set,
            args.aux_set,
            full_metrics,
        )

    save_dir = None
    if args.save:
        save_dir = hs_folder / "inference_plots_combined"
        save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for t0 in t0_values:
            idx_hs = t0 - hs_params["seq_len"]
            idx_shape = t0 - shape_params["seq_len"]

            X_hs, aux_hs, _ = hs_dataset[idx_hs]
            X_shape, aux_shape, _ = shape_dataset[idx_shape]
            X_hs, aux_hs = X_hs.unsqueeze(0).to(device), aux_hs.unsqueeze(0).to(device)
            X_shape, aux_shape = (
                X_shape.unsqueeze(0).to(device),
                aux_shape.unsqueeze(0).to(device),
            )

            hs_pred = hs_model.infer(
                X_hs, freqs, lead_time_steps, freq_means=hs_freq_means, aux=aux_hs
            )  # (1, L, 1) metres
            shape_pred = shape_model.infer(
                X_shape,
                freqs,
                lead_time_steps,
                freq_means=shape_freq_means,
                shape_means=shape_means,
                aux=aux_shape,
            )  # (1, L, F) log unit-area shape

            # exp() back to linear unit-area shape before recombining with m0.
            shape_pred_np = np.exp(shape_pred.cpu().numpy()[0])  # (L, F)
            m0_pred = ((hs_pred / 4.0) ** 2).cpu().numpy()[0]  # (L, 1)
            pred_phys = shape_pred_np * m0_pred  # (L, F)

            true_phys = test_density_phys.values[t0 : t0 + lead_time_steps]  # (L, F)
            pers_phys = np.tile(test_density_phys.values[t0 - 1], (lead_time_steps, 1))

            hs_pred_np = hs_pred.cpu().numpy()[0, :, 0]
            hs_pred_combined, tm02_pred = compute_bulk_params(
                pred_phys[np.newaxis], freqs_np
            )
            hs_true, tm02_true = compute_bulk_params(true_phys[np.newaxis], freqs_np)
            hs_pers, tm02_pers = compute_bulk_params(pers_phys[np.newaxis], freqs_np)
            hs_pred_combined, hs_true, hs_pers = (
                hs_pred_combined[0],
                hs_true[0],
                hs_pers[0],
            )
            tm02_pred, tm02_true = tm02_pred[0], tm02_true[0]

            # Sanity check: model.infer() renormalizes each predicted shape
            # step to unit area, so this should print ~1.000 for every step.
            # A drift here would mean the renormalization in
            # nn/transformer.py::infer() isn't being hit (e.g. a stale
            # checkpoint loaded against new code).
            shape_m0 = np.trapezoid(shape_pred_np, freqs_np, axis=1)
            print(f"\nSample t0={t0} (idx_hs={idx_hs}, idx_shape={idx_shape}):")
            print(
                f"  shape ∫ per step (should be ~1): "
                + ", ".join(f"{v:.3f}" for v in shape_m0)
            )
            print(
                f"  Hs from hs-model direct: "
                + ", ".join(f"{v:.3f}" for v in hs_pred_np)
            )
            print_bulk_table(
                t0,
                lead_time_steps,
                hs_true,
                hs_pred_combined,
                hs_pers,
                tm02_true,
                tm02_pred,
            )

            n_snap = min(args.n_steps, lead_time_steps)
            steps = sorted(
                set(np.linspace(0, lead_time_steps - 1, n_snap, dtype=int).tolist())
            )
            fig = plot_density_sample(
                t0,
                freqs_np,
                steps,
                pred_phys,
                true_phys,
                pers_phys,
                hs_pred_combined,
                hs_true,
                hs_pers,
                title_prefix="t0",
            )

            if save_dir is not None:
                out_path = save_dir / f"sample_t0_{t0}.png"
                fig.savefig(out_path, dpi=150)
                print(f"  Saved plot → {out_path}")

    plt.show()


def main():
    args = parse_args()
    if args.aggregate and args.target != "shape":
        raise ValueError("--aggregate is only supported with --target shape")
    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "buoy_data" / BUOY_ID / "processed_data.pkl"
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found — run scripts/data_processing.py first."
        )
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(file_path)
    freqs = get_freqs(density)
    freqs_np = freqs.numpy()
    device = get_device()

    if args.target == "combined":
        run_combined(
            args,
            project_root,
            density,
            alpha_1,
            alpha_2,
            r_1,
            r_2,
            wind,
            freqs,
            freqs_np,
            device,
        )
    else:
        run_single(
            args,
            project_root,
            density,
            alpha_1,
            alpha_2,
            r_1,
            r_2,
            wind,
            freqs,
            freqs_np,
            device,
        )


if __name__ == "__main__":
    main()
