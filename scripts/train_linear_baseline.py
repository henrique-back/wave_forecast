"""
Fit the final linear AR baseline (utils/linear_baseline.py) and evaluate it
on the held-out test set — the linear-baseline counterpart to
scripts/train.py's transformer retrain.

Reads results/linear_baseline/{target}/lead_{N}h/best_trial.txt if
scripts/optimize_linear_baseline.py has already been run (to pick up the
best order/ridge for this target/lead); otherwise falls back to
DEFAULT_ORDER/DEFAULT_RIDGE below, so this script also works standalone.

Writes a checkpoint (linear_baseline_final.pt) and metrics
(linear_baseline_metrics.json) into that same folder — the checkpoint is
what scripts/compare_versions.py --linear-baseline loads.

Run manually:
    python scripts/train_linear_baseline.py
"""
print("Importing packages")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ast
import json
import re
from pathlib import Path

import pandas as pd
import torch

from utils import get_freqs
from utils.linear_baseline import fit_linear_ar_from_density, evaluate_coeffs

print("Current working directory:", os.getcwd())

# ---------------------------------------------------------------------------
# Config — mirrors scripts/train.py's style: edit these constants and rerun.
# ---------------------------------------------------------------------------

BUOY_ID = "32012"
target = "shape"
lead_times_hours = [6, 12, 24, 48]

# Used only when results/linear_baseline/{target}/lead_{N}h/best_trial.txt
# doesn't exist yet (i.e. scripts/optimize_linear_baseline.py hasn't been run
# for this target/lead) — running that script first is optional.
DEFAULT_ORDER = 24
DEFAULT_RIDGE = 1e-6


def parse_best_trial(path: Path) -> dict:
    """Same ast.literal_eval approach scripts/train.py's parse_best_trial
    uses, adapted to scripts/optimize_linear_baseline.py's params dict
    (order, ridge instead of the transformer's hyperparameters)."""
    text = path.read_text()
    params_line = re.search(r"Best trial parameters:\n(.+)\n", text).group(1)
    return ast.literal_eval(params_line)


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

    for lead_time_hours in lead_times_hours:
        results_folder = (
            project_root / "results" / "linear_baseline" / target / f"lead_{lead_time_hours}h"
        )
        results_folder.mkdir(parents=True, exist_ok=True)

        best_trial_path = results_folder / "best_trial.txt"
        if best_trial_path.exists():
            params = parse_best_trial(best_trial_path)
            print(f"\n=== target={target!r} lead={lead_time_hours}h — "
                  f"using best_trial.txt params: {params} ===")
        else:
            params = {"order": DEFAULT_ORDER, "ridge": DEFAULT_RIDGE}
            print(f"\n=== target={target!r} lead={lead_time_hours}h — "
                  f"no best_trial.txt, using defaults: {params} ===")

        order, ridge = params["order"], params["ridge"]

        coeffs = fit_linear_ar_from_density(density, freqs, seq_len=order, target=target, ridge=ridge)
        val_metrics = evaluate_coeffs(density, freqs, coeffs, seq_len=order, lead_time=lead_time_hours,
                                       target=target, eval_split="val")
        test_metrics = evaluate_coeffs(density, freqs, coeffs, seq_len=order, lead_time=lead_time_hours,
                                        target=target, eval_split="test")

        ss_key = "Hs_SS" if target in ("hs", "density") else "Shape_SS"
        print(f"order={order} ridge={ridge} — "
              f"val {ss_key}: {val_metrics[ss_key]:.4f} | "
              f"test {ss_key}: {test_metrics[ss_key]:.4f}")

        checkpoint_path = results_folder / "linear_baseline_final.pt"
        torch.save(
            {
                "coeffs": coeffs,
                "order": order,
                "ridge": ridge,
                "target": target,
                "lead_time_steps": lead_time_hours,
                "freqs": freqs,
                "buoy_id": BUOY_ID,
            },
            checkpoint_path,
        )

        metrics_path = results_folder / "linear_baseline_metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "order": order,
                    "ridge": ridge,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                },
                indent=2,
            )
        )
        print(f"Saved checkpoint → {checkpoint_path}")
        print(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
