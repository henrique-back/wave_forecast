"""
Retrain a final model using the best hyperparameters found by scripts/optimize.py,
then evaluate it on the held-out test set (never touched during the Optuna search).

Reads results/{EXPERIMENT_NAME}/{target}/lead_{N}h/best_trial.txt for
each configured lead time, retrains from scratch with those hyperparameters
(train split only, early-stopped on val — same regime objective() used), and
writes a checkpoint + full metrics (val and test) back into that same folder.

Run manually:
    python scripts/train.py
"""

print("Importing packages")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import require_slurm
require_slurm("scripts/train.py")

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import get_freqs, set_seed, get_device
from nn import WaveHeightBaselineNN, evaluate
from nn.optimization import _prepare_dataloaders, _train_model
from nn.channels import CHANNEL_SETS, AUX_CHANNEL_SETS

print("Current working directory:", os.getcwd())

# ---------------------------------------------------------------------------
# Config — mirrors optimize.py's style: edit these constants and rerun.
# ---------------------------------------------------------------------------

# Must match an EXPERIMENT_NAME already produced by scripts/optimize.py —
# best_trial.txt is read from results/{EXPERIMENT_NAME}/{target}/lead_{N}h/.
# v9: matches optimize.py's STUDY_VERSION bump for the freq-axis conv
# padding fix — run optimize.py first to produce shape_v9's best_trial.txt.
EXPERIMENT_NAME = "shape_v12"
BUOY_ID = "32012"

target = "shape"
lead_times_hours = [12]

# Must match the CHANNEL_SET/AUX_SET that produced this experiment's
# best_trial.txt — see nn/channels.py and scripts/optimize.py.
CHANNEL_SET = "full"
AUX_SET = "dmd"
assert CHANNEL_SET in CHANNEL_SETS, f"CHANNEL_SET must be one of {list(CHANNEL_SETS)}"
assert AUX_SET in AUX_CHANNEL_SETS, f"AUX_SET must be one of {list(AUX_CHANNEL_SETS)}"

# Metric used to pick the best epoch during retraining. Should match the
# OBJECTIVE_METRIC that produced this experiment's best_trial.txt, so the
# retrained model is selected the same way the search selected it.
OBJECTIVE_METRIC = "Hs_SS" if target == "hs" else "final_step_SS_wasserstein"

# Seeds to retrain with. A single seed trains one final model. Add more to
# get a mean±std noise estimate across independent weight initializations and
# shuffle orders — every seed retrains from scratch on the same data with the
# same hyperparameters, so spread across seeds reflects training noise, not
# tuning quality.
SEEDS = [40, 41, 42, 43, 44]

NUM_EPOCHS = 100
PATIENCE = 20


def parse_best_trial(path: Path) -> dict:
    text = path.read_text()
    lead_hours = int(re.search(r"Lead time \(hours\): (\d+)", text).group(1))
    params_line = re.search(r"Best trial parameters:\n(.+)\n", text).group(1)
    params = ast.literal_eval(params_line)
    return {
        # optimize.py passes lead_time straight through as the step count
        # (no deltat downsampling wired into this script), so steps == hours.
        "lead_time_steps": lead_hours,
        "lead_time_hours": lead_hours,
        "params": params,
    }


def parse_current_best(path: Path, lead_time_hours: int) -> dict:
    """Fallback for a study that hasn't finished (no best_trial.txt yet).

    utils/save_progress.py overwrites current_best.txt after every trial, so
    it always reflects the best trial so far — but unlike best_trial.txt it
    has no 'Lead time (hours)' line, so lead_time_hours (the configured value
    that already selected this results_folder) is used directly instead.
    """
    text = path.read_text()
    params_line = re.search(r"Best params: (.+)\n", text).group(1)
    params = ast.literal_eval(params_line)
    return {
        "lead_time_steps": lead_time_hours,
        "lead_time_hours": lead_time_hours,
        "params": params,
    }


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Mean/std across seeds for every metric. Elementwise for per-step lists."""
    agg = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list]
        if isinstance(values[0], list):
            arr = np.array(values, dtype=float)  # (n_seeds, n_steps)
            agg[key] = {
                "mean": np.nanmean(arr, axis=0).tolist(),
                "std": np.nanstd(arr, axis=0).tolist(),
            }
        elif isinstance(values[0], (int, float)) or values[0] is None:
            arr = np.array(
                [v if v is not None else np.nan for v in values], dtype=float
            )
            agg[key] = {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}
        else:
            agg[key] = values
    return agg


def main():
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "buoy_data" / BUOY_ID / "processed_data.pkl"
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_path} not found — run scripts/data_processing.py first."
        )
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(file_path)
    print("Loaded preprocessed wave spectral data")

    freqs = get_freqs(density)
    device = get_device()
    print(f"Running on device: {device}")

    for lead_time_hours in lead_times_hours:
        results_folder = (
            project_root
            / "results"
            / EXPERIMENT_NAME
            / target
            / f"lead_{lead_time_hours}h"
        )
        best_trial_path = results_folder / "best_trial.txt"
        if best_trial_path.exists():
            trial_info = parse_best_trial(best_trial_path)
        else:
            current_best_path = results_folder / "current_best.txt"
            if not current_best_path.exists():
                print(f"no best_trial.txt or current_best.txt at {results_folder}")
                continue
            print(
                f"no best_trial.txt at {best_trial_path} — "
                f"falling back to {current_best_path}"
            )
            trial_info = parse_current_best(current_best_path, lead_time_hours)
        params = trial_info["params"]
        lead_time_steps = trial_info["lead_time_steps"]
        embed_dim = params["head_dim"] * params["nhead"]
        print(f"({lead_time_steps} steps) — params: {params} ===")

        val_metrics_per_seed = []
        test_metrics_per_seed = []

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")
            set_seed(seed)

            (
                train_loader,
                val_loader,
                test_loader,
                freq_means,
                shape_means,
                num_freqs,
                num_channels,
                num_aux_channels,
            ) = _prepare_dataloaders(
                density,
                alpha_1,
                alpha_2,
                r_1,
                r_2,
                params["seq_len"],
                lead_time_steps,
                params["batch_size"],
                target,
                shuffle_seed=seed,
                wind=wind,
                channel_set=CHANNEL_SET,
                aux_set=AUX_SET,
            )

            model = WaveHeightBaselineNN(
                num_freqs=num_freqs,
                freqs=freqs,
                target=target,
                num_channels=num_channels,
                num_aux_channels=num_aux_channels,
                freq_embed_dropout=params["freq_embed_dropout"],
                embed_dropout=params["embed_dropout"],
                nhead=params["nhead"],
                num_encoder_layers=params["num_encoder_layers"],
                num_decoder_layers=params["num_decoder_layers"],
                embed_dim=embed_dim,
            ).to(device)

            best_val_score, best_val_metrics, best_model_state = _train_model(
                model,
                train_loader,
                val_loader,
                device,
                freqs,
                freq_means,
                shape_means,
                target,
                lead_time_steps,
                params["lr"],
                params["weight_decay"],
                OBJECTIVE_METRIC,
                num_epochs=NUM_EPOCHS,
                patience=PATIENCE,
                trial=None,
                # Now a tuned hyperparameter (nn/optimization.py::objective,
                # optimize.py v12+) rather than a manually-set constant, so
                # it's read from best_trial.txt like every other param.
                # .get(..., 0.0) falls back to the pre-v12 no-op behavior for
                # older best_trial.txt files that predate this hyperparameter.
                wasserstein_loss_weight=params.get("wasserstein_loss_weight", 0.0),
                # Not yet in objective()'s search space (see
                # nn/optimization.py::_train_model's docstring) — best_trial.txt
                # will never actually contain this key until a Stage 2
                # promotion, so .get(..., 0.0) is a no-op here today. Kept for
                # forward-compatibility with the same convention
                # wasserstein_loss_weight uses, and so a manually-edited
                # best_trial.txt (or a direct _train_model(...) call bypassing
                # this script) can already override it for a Stage 1 A/B sweep.
                kl_loss_weight=params.get("kl_loss_weight", 0.0),
            )

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            test_metrics = evaluate(
                model,
                test_loader,
                device,
                freqs,
                lead_time=lead_time_steps,
                freq_means=freq_means,
                shape_means=shape_means,
            )
            print(
                f"Seed {seed} — best val {OBJECTIVE_METRIC}: {best_val_score:.4f} | "
                f"test RMSE: {test_metrics['RMSE']:.4f} | "
                f"test overall_SS: {test_metrics['overall_SS']:.4f}"
            )

            val_metrics_per_seed.append(best_val_metrics)
            test_metrics_per_seed.append(test_metrics)

            checkpoint_path = results_folder / f"final_model_seed{seed}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "params": params,
                    "target": target,
                    "lead_time_steps": lead_time_steps,
                    "lead_time_hours": lead_time_hours,
                    "seed": seed,
                    "freq_means": freq_means,
                    "shape_means": shape_means,
                    "freqs": freqs,
                },
                checkpoint_path,
            )

            metrics_path = results_folder / f"final_metrics_seed{seed}.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "params": params,
                        "val_metrics": best_val_metrics,
                        "test_metrics": test_metrics,
                    },
                    indent=2,
                )
            )
            print(f"Saved checkpoint → {checkpoint_path}")
            print(f"Saved metrics → {metrics_path}")

        if len(SEEDS) > 1:
            summary = {
                "seeds": SEEDS,
                "params": params,
                "val_metrics": _aggregate_metrics(val_metrics_per_seed),
                "test_metrics": _aggregate_metrics(test_metrics_per_seed),
            }
            summary_path = results_folder / "final_metrics_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"Saved {len(SEEDS)}-seed summary → {summary_path}")


if __name__ == "__main__":
    main()
