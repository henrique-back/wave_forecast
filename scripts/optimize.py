print("Importing packages")
import sys
import os
import subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
# changes in a way that makes old trials incomparable.  A new version creates
# a fresh study (and fresh DB file) so stale trials never corrupt the TPE
# surrogate model.
STUDY_VERSION = "v5"

# Short slug used as the top-level folder under results/.
# Change this whenever you start a new experiment (new architecture, new
# input variables, etc.) so that each run's results are stored separately
# and can be compared in RESEARCH_LOG.md.
# Convention: {short_description}_{STUDY_VERSION}  e.g. 'freq_embedding_v3'
EXPERIMENT_NAME = "hs_shape_v5"

# Human-readable description written once to results/{EXPERIMENT_NAME}/metadata.md.
EXPERIMENT_DESCRIPTION = (
    "Transformer with convolutional frontend and frequency-structured embedding."
    "Uses weighted mean Skill Score as objective."
    "Trains to predict Hs and spectral shape (density target) at 6h, 12h, 24h, 48h lead times."
)

# Set parameters
lead_times_hours = [6, 12, 24, 48]
target = "shape"
n_trials = 50

# Which frequency-resolved channels feed the encoder. See nn/channels.py.
#   'density' : spectral density only
#   'full'    : density + alpha_1 + alpha_2 + r_1 (current default)
CHANNEL_SET = "full"
assert CHANNEL_SET in CHANNEL_SETS, f"CHANNEL_SET must be one of {list(CHANNEL_SETS)}"

# Which scalar side-input (aux) channels are fused into the encoder. See
# nn/channels.py. 'wind' requires buoy_data/wind.txt to have been processed
# (i.e. processed_data.pkl regenerated after utils/data_processing.py added
# wind support).
#   'none' : no auxiliary input (current default)
#   'wind' : wind_u/wind_v
AUX_SET = "none"
assert AUX_SET in AUX_CHANNEL_SETS, f"AUX_SET must be one of {list(AUX_CHANNEL_SETS)}"

# Metric used to select the best epoch, drive early stopping and LR scheduling,
# and report the Optuna trial value.  Must be one of:
#   'weighted_mean_SS'  exponentially-weighted mean per-step Skill Score (default)
#   'overall_SS'        Skill Score on flattened all-step RMSE
#   'RMSE'              negative overall RMSE
#   'Hs_RMSE'           negative Hs RMSE           (density target only)
#   'Tm02_RMSE'         negative Tm02 RMSE          (density target only)
#   'Shape_RMSE'        negative spectral shape RMSE (density target only)
#   'SI_mean'           negative mean Scatter Index  (density target only)
OBJECTIVE_METRIC = "weighted_mean_SS"

# Process data
project_root = Path(__file__).resolve().parent.parent
folder_path = project_root / "buoy_data"
file_path = folder_path / "processed_data.pkl"

# Load from file if it exists
if file_path.exists():
    dfs_interpolated = pd.read_pickle(file_path)
    density, alpha_1, alpha_2, r_1, wind = dfs_interpolated
    print("Loaded preprocessed wave spectral data")
else:
    from utils.data_processing import data_processing  # or wherever your function lives

    density, alpha_1, alpha_2, r_1, wind = data_processing(
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
        f"- **Architecture**: (fill in manually)\n"
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
        wind=wind,
        channel_set=CHANNEL_SET,
        aux_set=AUX_SET,
        freqs=freqs,
        lead_time=lead_time_hours,
        target=target,
        objective_metric=OBJECTIVE_METRIC,
        results_folder=results_folder,
    )

    # Run optuna
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=20, multivariate=True, seed=42
    )
    # num_epochs=100, early stopping patience=10 → ~20% warmup = 20 steps
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///optuna_study_{STUDY_VERSION}.db",
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        callbacks=[lambda study, trial: save_progress(study, trial, results_folder)],
        # A single trial hitting CUDA OOM (e.g. a large batch_size /
        # lead_time / embed_dim combination) must not take down the
        # whole multi-hour study — mark it failed and keep going.
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
                "val_Shape_RMSE",
                "val_Shape_masked_samples",
                "val_SI_mean",
            ]:
                if key in attrs:
                    f.write(f"{key}: {attrs[key]}\n")
            if "val_SI_per_bin" in attrs:
                f.write(f"val_SI_per_bin: {attrs['val_SI_per_bin']}\n")

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
