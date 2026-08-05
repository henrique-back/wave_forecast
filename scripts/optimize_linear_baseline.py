"""
Grid-search the linear AR baseline's order (AR lag length) and ridge,
scored on the validation split, and write results/linear_baseline/{target}/
lead_{N}h/best_trial.txt so scripts/train_linear_baseline.py can pick up the
winning config for a final train-fit + held-out test evaluation.

Unlike scripts/optimize.py's Optuna search over the transformer's
hyperparameters, fitting utils/linear_baseline.py's AR model is a
closed-form least-squares solve (no training loop, no randomness), so a
small brute-force grid over ORDER_CANDIDATES x RIDGE_CANDIDATES is cheap
enough to just enumerate directly rather than reaching for Optuna.

Running this script first is optional — scripts/train_linear_baseline.py
falls back to its own DEFAULT_ORDER/DEFAULT_RIDGE constants when no
best_trial.txt exists yet.

Run manually:
    python scripts/optimize_linear_baseline.py
"""
print("Importing packages")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
from pathlib import Path

import pandas as pd

from utils import get_freqs
from utils.linear_baseline import evaluate_linear_ar

print("Current working directory:", os.getcwd())

# ---------------------------------------------------------------------------
# Config — mirrors scripts/optimize.py's style: edit these constants and rerun.
# ---------------------------------------------------------------------------

BUOY_ID = "32012"
target = "shape"
lead_times_hours = [6, 12, 24, 48]

# Same categorical seq_len choices nn/optimization.py::objective() searches
# for the transformer, so the linear baseline gets a fair shot at whichever
# lookback window suits it best, independent of what the transformer landed
# on for the same target/lead.
ORDER_CANDIDATES = [12, 24, 48, 96]

# fit_linear_ar's ridge parameter exists purely for numerical stability
# (adjacent lags/frequency bins are highly correlated), not meaningful
# shrinkage — see its docstring. Kept as a list for extensibility, but one
# tiny value is enough by default.
RIDGE_CANDIDATES = [1e-6]

# Mirrors scripts/train.py's own default choice of objective metric.
OBJECTIVE_METRIC = "Hs_SS" if target == "hs" else "weighted_mean_SS"


def _weighted_mean_ss(per_step_ss):
    """Exponentially-weighted mean Skill Score — same formula as
    nn/optimization.py::_weighted_mean_ss, duplicated here (rather than
    imported) so this script doesn't need to pull in the whole Optuna/
    training module for a two-line formula."""
    n = len(per_step_ss)
    half_life = max(1.0, n / 2.0)
    weights = [math.exp(-t * math.log(2) / half_life) for t in range(n)]
    weight_sum = sum(weights)
    return sum(w * ss for w, ss in zip(weights, per_step_ss)) / weight_sum


def _score(metrics, objective_metric):
    if objective_metric == "Hs_SS":
        return metrics["Hs_SS"]
    elif objective_metric == "weighted_mean_SS":
        return _weighted_mean_ss(metrics["per_step_SS"])
    else:
        raise ValueError(f"Unknown OBJECTIVE_METRIC {objective_metric!r}")


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
        print(f"\n=== Linear AR baseline: target={target!r} lead={lead_time_hours}h ===")
        results_folder = (
            project_root / "results" / "linear_baseline" / target / f"lead_{lead_time_hours}h"
        )
        results_folder.mkdir(parents=True, exist_ok=True)

        best = None
        for order in ORDER_CANDIDATES:
            for ridge in RIDGE_CANDIDATES:
                val_metrics = evaluate_linear_ar(
                    density, freqs, seq_len=order, lead_time=lead_time_hours,
                    target=target, ridge=ridge, eval_split="val",
                )
                score = _score(val_metrics, OBJECTIVE_METRIC)
                print(f"  order={order:>3} ridge={ridge:.0e}: {OBJECTIVE_METRIC}={score:.4f}")
                if best is None or score > best["score"]:
                    best = {"order": order, "ridge": ridge, "score": score, "val_metrics": val_metrics}

        print(f"Best: order={best['order']} ridge={best['ridge']} "
              f"{OBJECTIVE_METRIC}={best['score']:.4f}")

        result_file = results_folder / "best_trial.txt"
        result_file.write_text(
            f"Lead time (hours): {lead_time_hours}\n"
            f"Best trial parameters:\n"
            f"{{'order': {best['order']}, 'ridge': {best['ridge']}}}\n"
            f"Best {OBJECTIVE_METRIC}: {best['score']}\n"
            f"val_metrics: {best['val_metrics']}\n"
        )
        print(f"Saved {result_file}")


if __name__ == "__main__":
    main()
