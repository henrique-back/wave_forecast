print("Importing packages")
import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import require_slurm
require_slurm("scripts/optimize.py")

from pathlib import Path
import pandas as pd
import torch
import optuna
import optuna.visualization as vis
from utils import get_freqs, set_seed, data_processing, save_progress
from nn import objective
from nn.channels import CHANNEL_SETS, AUX_CHANNEL_SETS
from functools import partial

print("Current working directory:", os.getcwd())

# Set randomness seed
set_seed(42)

# Bump this when objective definition, hyperparameter space, or training logic
# changes in a way that makes old trials incomparable. A new version creates
# a fresh study (and fresh DB file) so stale trials never corrupt the TPE
# surrogate model. Full history of what changed at each version and why:
# manuscript/decisions/README.md (entries 010-027 cover v8-v13).
STUDY_VERSION = "v13"

# Short slug used as the top-level folder under results/.
# Change this whenever you start a new experiment (new architecture, new
# input variables, etc.) so that each run's results are stored separately
# and can be compared in RESEARCH_LOG.md.
# Convention: {short_description}_{STUDY_VERSION}  e.g. 'freq_embedding_v3'
EXPERIMENT_NAME = "shape_v13"

# Human-readable description written once to results/{EXPERIMENT_NAME}/metadata.md.
EXPERIMENT_DESCRIPTION = (
    "Transformer with convolutional frontend and frequency-structured embedding."
    "Implements attention pooling"
    "Uses final-step Skill Score as objective (last forecast step only, not "
    "an average across autoregressive steps)."
    "Trains to predict spectral shape at 6h, 12h, 24h lead times."
    "Fixes RMSE weighting to use utils.trapz_weights instead of flat mean over log-spaced frequency grid."
    "Adds padding_mode to convolutional frontend."
    "Includes r2 as new channel."
    "Switches optimizer to AdamW, narrows lr search range around shape_v9's best trials, and "
    "splits the single dropout hyperparameter into freq_embed_dropout and embed_dropout (the "
    "latter now also drives nn.Transformer's own internal dropout, previously unwired)."
    "metric computed only on lead time step of interest, not averaged across autoregressive steps."
    "v11: predicts log-spectral-energy (log E(f)/m0) directly via a plain "
    "Linear head instead of a Softplus-activated linear value; loss switched "
    "to frequency-weighted plain MSE in log-space; non-negativity of the "
    "physical shape now comes from exp() at inference/metric time instead of "
    "an architectural Softplus constraint."
    "v12: adds a tunable auxiliary Wasserstein-distance loss term "
    "(wasserstein_loss_weight, see utils/loss.py::SpectralWassersteinLoss) "
    "targeting multimodal (double/triple-peaked) sea states that a "
    "whole-spectrum frequency-weighted loss blurs into one smoothed hump, "
    "and switches AUX_SET to 'dmd' — Dynamic Mode Decomposition features "
    "(nn/prepare_dmd.py) giving the encoder each sample's dominant "
    "growth/decay rate and oscillation frequency from its input window, "
    "instead of requiring the model to infer current wave-system dynamics "
    "implicitly. Both validated by a manual before/after comparison prior to "
    "being added to the search space — see STUDY_VERSION's v12 comment above."
    "v13: promotes the KL/Wasserstein/Peak composite loss validated by "
    "scripts/ablate_loss.py's small fixed-architecture ablation "
    "(STUDY_VERSION lossablation_v2) into the real search space — "
    "base_loss_weight=0 (literal substitute of the per-bin loss), "
    "L = kl_loss_weight*D_KL + wasserstein_loss_weight*W2 + peak_loss_weight*"
    "L_peak, all three now tunable. OBJECTIVE_METRIC switches to "
    "peak_fidelity_SS accordingly — see STUDY_VERSION's v13 comment above "
    "for the full rationale and results/lossablation_comparison_v2.md for "
    "the ablation's own numbers."
)

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--lead", type=int, default=None, choices=[6, 12, 24, 48],
                      help="Run a single lead time (for one-lead-per-Slurm-job "
                           "submission, e.g. slurm/optimize_v13_lead*.slurm) instead "
                           "of looping over lead_times_hours in one process.")
_args, _ = _parser.parse_known_args()

# Set parameters
lead_times_hours = [_args.lead] if _args.lead is not None else [12, 24, 48]
target = "shape"

# target == 'shape' only; pass fixed_head_dim=None/fixed_nhead=None to
# nn.objective to go back to searching them. See manuscript/decisions/log/027.
FIXED_HEAD_DIM = 32
FIXED_NHEAD = 8

# 11 tunable hyperparameters (2 categorical, 2 int, 7 continuous).
# n_startup_trials=18 random samples for multivariate TPE's initial KDE;
# n_trials=80 leaves 62 for TPE to exploit it. Each of the 3 lead times
# (--lead) runs this budget as its own Slurm job.
n_trials = 80

# Which frequency-resolved channels feed the encoder. See nn/channels.py.
#   'density' : spectral density only
#   'full'    : density + alpha_1 + alpha_2 + r_1 + r_2 (current default)
CHANNEL_SET = "full"
assert CHANNEL_SET in CHANNEL_SETS, f"CHANNEL_SET must be one of {list(CHANNEL_SETS)}"

# Which scalar side-input (aux) channels are fused into the encoder. See
# nn/channels.py. 'wind' requires buoy_data/wind.txt to have been processed
# (i.e. processed_data.pkl regenerated after utils/data_processing.py added
# wind support).
#   'none' : no auxiliary input
#   'wind' : wind_u/wind_v
#   'dmd'  : Dynamic Mode Decomposition growth-rate/frequency/amplitude
#            features from the input window's density history
#            (nn/prepare_dmd.py; manuscript/decisions/log/019)
AUX_SET = "dmd"
assert AUX_SET in AUX_CHANNEL_SETS, f"AUX_SET must be one of {list(AUX_CHANNEL_SETS)}"

# Metric used to select the best epoch, drive early stopping and LR scheduling,
# and report the Optuna trial value. See nn/optimization.py::_compute_val_score
# for the full definition of each; manuscript/decisions/log/{001,009,014,021,026}
# for why each was introduced. Must be one of: 'final_step_SS', 'weighted_mean_SS',
# 'overall_SS', 'Hs_SS', 'RMSE', 'Hs_RMSE', 'Tm02_RMSE', 'Shape_RMSE', 'SI_mean',
# 'final_step_SS_wasserstein', 'peak_fidelity_SS' (target=='shape' only, requires
# compute_peak_metrics=True — see COMPUTE_PEAK_METRICS below).
OBJECTIVE_METRIC = "Hs_SS" if target == "hs" else "peak_fidelity_SS"

# Assembles scripts/ablate_loss.py's validated 'combined' recipe (manuscript/
# decisions/log/026) — target=='shape' only, since that's the ablation's
# validated scope; base_loss_weight=1.0 (per-bin loss, unaffected) otherwise.
BASE_LOSS_WEIGHT = 0.0 if target == "shape" else 1.0
COMPUTE_PEAK_METRICS = (target == "shape")

# Process data
BUOY_ID = "32012"
project_root = Path(__file__).resolve().parent.parent
folder_path = project_root / "buoy_data" / BUOY_ID
file_path = folder_path / "processed_data.pkl"

# Load from file if it exists
if file_path.exists():
    dfs_interpolated = pd.read_pickle(file_path)
    density, alpha_1, alpha_2, r_1, r_2, wind = dfs_interpolated
    print("Loaded preprocessed wave spectral data")
else:
    from utils.data_processing import data_processing  # or wherever your function lives

    density, alpha_1, alpha_2, r_1, r_2, wind = data_processing(
        folder_path, save_path=file_path
    )

freqs = get_freqs(density)

# Write experiment metadata once (idempotent — safe to re-run)
_experiment_dir = Path(__file__).parent.parent / "results" / EXPERIMENT_NAME
_experiment_dir.mkdir(parents=True, exist_ok=True)
_meta_path = _experiment_dir / "metadata.md"
if not _meta_path.exists():
    _meta_path.write_text(
        f"# Experiment: {EXPERIMENT_NAME}\n\n"
        f"- **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n"
        f"- **Description**: {EXPERIMENT_DESCRIPTION}\n"
        f"- **STUDY_VERSION**: {STUDY_VERSION}\n"
        f"- **OBJECTIVE_METRIC**: {OBJECTIVE_METRIC}\n"
        f"- **CHANNEL_SET**: {CHANNEL_SET}\n"
        f"- **AUX_SET**: {AUX_SET}\n"
        f"- **BASE_LOSS_WEIGHT**: {BASE_LOSS_WEIGHT}\n"
        f"- **FIXED_HEAD_DIM / FIXED_NHEAD**: {FIXED_HEAD_DIM} / {FIXED_NHEAD} "
        f"(shape target only; see manuscript/decisions/log/027)\n"
        f"- **Architecture**: (fill in manually)\n"
    )


# The 12h/24h/48h studies share one optuna_study_{STUDY_VERSION}.db file and
# are commonly launched as separate concurrent processes; timeout=30 avoids
# lock contention errors under Python sqlite3's default 5s busy-timeout.
# Deliberately not WAL mode — see manuscript/decisions/log/018.
storage = optuna.storages.RDBStorage(
    url=f"sqlite:///optuna_study_{STUDY_VERSION}.db",
    engine_kwargs={"connect_args": {"timeout": 30}},
)

for lead_time_hours in lead_times_hours:
    print(f"\n=== Optimizing for lead_time={lead_time_hours}h ===")

    # Create unique Optuna study name — channel_set/aux_set are included so
    # a study never silently mixes trials across incompatible input configs.
    target_folder = f"{target}"
    lead_hours = f"lead_{lead_time_hours}h"
    study_name = f"{target_folder}_{CHANNEL_SET}_{AUX_SET}_{lead_hours}_{STUDY_VERSION}"

    # Folder for results — nested under the experiment name
    results_folder = (
        Path(__file__).parent.parent
        / "results"
        / EXPERIMENT_NAME
        / target_folder
        / lead_hours
    )
    results_folder.mkdir(parents=True, exist_ok=True)

    # Define objective function
    objective_fn = partial(
        objective,
        density=density,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        r_1=r_1,
        r_2=r_2,
        wind=wind,
        channel_set=CHANNEL_SET,
        aux_set=AUX_SET,
        freqs=freqs,
        lead_time=lead_time_hours,
        target=target,
        objective_metric=OBJECTIVE_METRIC,
        results_folder=results_folder,
        base_loss_weight=BASE_LOSS_WEIGHT,
        compute_peak_metrics=COMPUTE_PEAK_METRICS,
        fixed_head_dim=FIXED_HEAD_DIM if target == "shape" else None,
        fixed_nhead=FIXED_NHEAD if target == "shape" else None,
    )

    # Run optuna
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=18, multivariate=True, seed=42
    )
    # See manuscript/decisions/log/005 (n_warmup_steps) and /015 (PatientPruner).
    median_pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=30, n_min_trials=5, interval_steps=5
    )
    pruner = optuna.pruners.PatientPruner(
        median_pruner, patience=10, min_delta=0.0
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        callbacks=[lambda study, trial: save_progress(study, trial, results_folder)],
        catch=(torch.OutOfMemoryError,),
    )
    print("Best trial:")
    print(study.best_trial.params)
    print("Validation loss:", study.best_value)

    result_file = os.path.join(results_folder, "best_trial.txt")
    with open(result_file, "w") as f:
        f.write(f"Lead time (hours): {lead_time_hours}\n")
        f.write("Best trial parameters:\n")
        f.write(str(study.best_trial.params) + "\n")
        attrs = study.best_trial.user_attrs
        scalar_keys = [
            "val_RMSE",
            "val_MAPE",
            "val_CC",
            "val_Bias",
            "val_R2",
            "val_overall_SS",
        ]
        list_keys = [
            "val_per_step_RMSE",
            "val_per_step_RMSE_pers",
            "val_per_step_SS",
            "val_per_step_Bias",
            "val_per_step_R2",
        ]
        for key in scalar_keys:
            if key in attrs:
                f.write(f"{key}: {attrs[key]}\n")
        for key in list_keys:
            if key in attrs:
                f.write(f"{key}: {attrs[key]}\n")
        if target == "density":
            for key in [
                "val_Hs_RMSE",
                "val_Hs_Bias",
                "val_Tm02_RMSE",
                "val_Tm02_Bias",
                "val_Shape_masked_samples",
                "val_SI_mean",
            ]:
                if key in attrs:
                    f.write(f"{key}: {attrs[key]}\n")
            if "val_SI_per_bin" in attrs:
                f.write(f"val_SI_per_bin: {attrs['val_SI_per_bin']}\n")
        if target in ("density", "shape"):
            # Shape_RMSE/Shape_SS are computed for both target types (see
            # nn/evaluate.py); Shape_Mass_Error only for 'shape'.
            for key in ["val_Shape_RMSE", "val_Shape_SS", "val_Shape_Mass_Error"]:
                if key in attrs:
                    f.write(f"{key}: {attrs[key]}\n")

    print(f"Results saved to {result_file}")

    # Save visualizations
    fig = vis.plot_param_importances(study)
    fig.write_html(os.path.join(results_folder, "param_importances.html"))

    fig = vis.plot_optimization_history(study)
    fig.write_html(os.path.join(results_folder, "optimization_history.html"))

    print(f"Visualizations saved to {results_folder}")

# Regenerate the research log after all studies finish
_summarize = Path(__file__).parent / "summarize_results.py"
subprocess.run([sys.executable, str(_summarize)], check=False)
