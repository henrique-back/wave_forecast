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
#
# Bumped v7 -> v8: RMSE (both the training loss for 'density'/'shape'
# targets, RMSELoss in utils/loss.py, and the RMSE/CC/Bias/R2/SS family in
# nn/evaluate.py + nn/spectrum_eval.py) is now frequency-weighted via
# utils.trapz_weights instead of a flat mean over the log-spaced frequency
# grid. This changes both what the model optimizes for and how trials are
# scored, so v7 trials are not comparable to v8 trials.
#
# v9: FreqDimEmbedding's frequency-axis conv (nn/freq_embedding.py's
# freq_conv, reusing TemporalConvFrontend) switched from zero-padding to
# replicate-padding at the grid boundary. Zero-padding fabricated fake
# zero-energy bins just outside 0.02-0.485 Hz and corrupted the ~7 bins
# nearest each edge (visible as a spurious low-frequency bump/negative dip
# in shape_v8's test-set predictions) — an architecture change, so v8 trials
# (optuna_study_v8.db) are not comparable to v9 trials either.
#
# v10 (in place, no version bump — the original v10 DB was deleted before any
# of these trials were compared against): switched the optimizer from
# optim.Adam to optim.AdamW (nn/optimization.py::_train_model), narrowed lr's
# search range to bracket shape_v9's best trials (1e-3 - 1.5e-2, was
# 1e-4 - 1e-2), and split the single `dropout` hyperparameter into
# `freq_embed_dropout` (FreqDimEmbedding's freq_embed_dim=8-wide internal
# conv) and `embed_dropout` (PositionalEncoding, the top-level time-axis
# TemporalConvFrontend, and nn.Transformer's own internal attention/FFN
# dropout — this last one was previously never wired up at all and silently
# stuck at PyTorch's default of 0.1). weight_decay's range is intentionally
# left unchanged despite the optimizer switch: Adam's coupled L2 weight decay
# and AdamW's decoupled weight decay behave differently for the same numeric
# value, so shape_v9's Adam-tuned weight_decay values aren't known to
# transfer. See nn/optimization.py::objective() for the exact ranges.
#
# v11: OBJECTIVE_METRIC (non-'hs' targets) switched from 'weighted_mean_SS' to
# 'final_step_SS' (nn/optimization.py::_compute_val_score) — the exponential
# per-step weighting biased trial/checkpoint selection toward the earlier,
# easier autoregressive steps rather than the step that's actually the
# forecast product at this lead time (intermediate steps are just
# autoregressive scaffolding to get there). This changes what "best" means
# for the same trial, so v10 trials are not comparable to v11 trials.
#
# v12: 'density'/'shape' targets now predict log-spectral-energy directly
# (log E(f) / log E(f)/m0) instead of a Softplus-activated non-negative
# linear value — see nn/transformer.py's predictor construction, and the
# log-space transform applied to y_batch in nn/training_loop.py/nn/evaluate.py
# (utils/log_transform.py::to_log_space). The training loss for these two
# targets also switched from frequency-weighted RMSE to frequency-weighted
# plain MSE, computed in log-space. This is an incompatible objective-function
# change: v11 trials are not comparable to v12 trials. Two effects worth
# flagging when actually running this study:
#   - lr's search range (nn/optimization.py::objective(), currently
#     1e-3-1.5e-2) was narrowed to bracket shape_v9's best trials under the
#     OLD Softplus + physical-space-RMSE regime — same situation this file's
#     own v10 comment already flagged for weight_decay after the Adam->AdamW
#     switch. Consider widening it back out rather than assuming it still
#     applies.
#   - OBJECTIVE_METRIC choices derived from per_step_SS/overall_SS
#     ('final_step_SS', 'weighted_mean_SS', 'overall_SS') are now computed in
#     log-space for 'density'/'shape' and are NOT comparable to pre-v12 runs.
#     'Hs_SS', 'Tm02_RMSE', and 'Shape_RMSE' are exp()'d back to physical
#     units internally (nn/evaluate.py) and remain the fair basis for
#     comparing this ablation against pre-v12 results.
#
# NOTE on the v11 -> v12 gap: the paragraph above was apparently written when
# the log-spectral-energy / log-space-MSE change shipped, but STUDY_VERSION
# itself was left at "v11" — no optuna_study_v12.db or results/shape_v12/
# were ever produced under that label, yet shape_v11's actual results already
# reflect this log-space behavior (current nn/training_loop.py code). So the
# version bump for that change was seemingly never applied even though the
# code and this comment shipped. Left as-is (not retroactively relabeled) —
# "v12" below is the next genuinely unused version number, used for a
# different, later change.
#
# v12: two additions, both validated by a manual before/after comparison
# (not committed — reusing shape_v11's exact other hyperparameters) before
# being wired into the search space here:
#   - wasserstein_loss_weight (nn/optimization.py::objective(), new
#     trial.suggest_float hyperparameter): an auxiliary SpectralWassersteinLoss
#     term (utils/loss.py) for target=='shape' — at the time this was
#     written, the 1-D Wasserstein-1 earth-mover distance between predicted
#     and true spectra (exact via CDF L1 distance), added to the existing
#     per-bin loss. Manually swept at weights 50/150 vs a 0-weight control:
#     every metric improved monotonically (Shape_RMSE 2.244->2.083, Shape_SS
#     0.107->0.171, Peak_Separation_Recall 0.580->0.617 at weight 150) and
#     the improvement was confirmed visually sharper on known-multimodal
#     test samples, not just numerically better — see nn/training_loop.py's
#     docstring for why this term exists (a whole-spectrum frequency-weighted
#     loss gives multimodal peaks no special treatment on its own; W1 pushes
#     back on "too flat" specifically).
#     NOTE (2026-08-17): SpectralWassersteinLoss was subsequently switched
#     from W1 to W2 (quadratic transport cost, quantile-domain formula —
#     see utils/loss.py's docstring) as part of the KL/Wasserstein/peak loss
#     ablation. The manually-swept weights (50/150) and the 10-400 search
#     range below were tuned under the OLD W1 metric's scale; W2's values
#     are not known to share the same numeric scale (W1's ∫|CDF gap|df and
#     W2's sqrt(∫(quantile gap)^2 dq) are different quantities, not just a
#     rescaling of each other), so treat any live v12 trial completed before
#     this date as using a different, incomparable wasserstein_loss_weight
#     definition than trials completed after it.
#   - AUX_SET switched from 'none' to 'dmd' (see below): Dynamic Mode
#     Decomposition features (nn/prepare_dmd.py) computed per-sample from the
#     encoder's input window of (already-normalized) density spectra — a few
#     dominant modes' growth/decay rate and oscillation frequency, giving the
#     encoder direct information about whether currently-observed wave
#     systems are growing or decaying, rather than requiring the Transformer
#     to infer that implicitly. Manually validated alone (smaller, less
#     consistent effect than the Wasserstein term) and in combination with it
#     (best Peak_Separation_Recall/Peak_Count_Pred_Mean of anything tested,
#     and the visually sharpest/tallest peaks across every known-multimodal
#     sample checked, even though whole-spectrum Shape_RMSE slightly favored
#     the Wasserstein-only run — the two metrics disagreeing here is itself
#     consistent with this project's recurring finding that aggregate
#     spectrum-wide error metrics dilute peak-specific behavior).
#   - OBJECTIVE_METRIC switched from 'final_step_SS' to
#     'final_step_SS_wasserstein' (nn/optimization.py::_compute_val_score):
#     training now includes the Wasserstein term above, so selecting
#     epochs/trials by plain final_step_SS (blind to Shape_Wasserstein) would
#     be inconsistent with what's actually being optimized for — risking
#     silently discarding a better-separated-peaks checkpoint in favor of one
#     that's marginally better on a metric known to dilute exactly that
#     property. The blend uses a fixed constant weight
#     (_FINAL_STEP_SS_WASSERSTEIN_BETA), not the trial's own
#     wasserstein_loss_weight, to keep cross-trial comparison fair.
STUDY_VERSION = "v12"

# Short slug used as the top-level folder under results/.
# Change this whenever you start a new experiment (new architecture, new
# input variables, etc.) so that each run's results are stored separately
# and can be compared in RESEARCH_LOG.md.
# Convention: {short_description}_{STUDY_VERSION}  e.g. 'freq_embedding_v3'
EXPERIMENT_NAME = "shape_v12"

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
)

# Set parameters
lead_times_hours = [48]
target = "shape"
# With 10 tunable hyperparameters (4 categorical, 2 int, 4 continuous —
# wasserstein_loss_weight added in v12), n_startup_trials=15 gives
# multivariate TPE enough random samples to fit an
# initial KDE without eating half the budget on pure random search (as
# n_startup_trials=20 of n_trials=40 did previously); n_trials=80 leaves 65
# trials for TPE to actually exploit that model, vs. only 20 before.
n_trials = 70

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
#            features from the input window's density history (nn/prepare_dmd.py)
#            — current default as of v12, see STUDY_VERSION's changelog comment.
AUX_SET = "dmd"
assert AUX_SET in AUX_CHANNEL_SETS, f"AUX_SET must be one of {list(AUX_CHANNEL_SETS)}"

# Metric used to select the best epoch, drive early stopping and LR scheduling,
# and report the Optuna trial value.  Must be one of:
#   'final_step_SS'     Skill Score at the last forecast step only — the
#                       actual chosen lead time, since the intermediate
#                       autoregressive steps are scaffolding, not a deliverable
#   'weighted_mean_SS'  exponentially-weighted mean per-step Skill Score
#                       (biases toward earlier/easier steps, not the step
#                       that's actually forecast)
#   'overall_SS'        Skill Score on flattened all-step RMSE
#   'Hs_SS'             Hs Skill Score — robust to seq_len; use when Hs accuracy
#                       is the primary goal. For target=='hs' equals overall_SS;
#                       for target=='density' computed from denormalised spectra.
#   'RMSE'              negative overall RMSE
#   'Hs_RMSE'           negative Hs RMSE           (density target only)
#   'Tm02_RMSE'         negative Tm02 RMSE          (density target only)
#   'Shape_RMSE'        negative spectral shape RMSE (density target only)
#   'SI_mean'           negative mean Scatter Index  (density target only)
#   'final_step_SS_wasserstein'  final_step_SS minus a FIXED penalty on
#                       Shape_Wasserstein (works for both 'shape' and
#                       'density' targets — see nn/evaluate.py, which
#                       computes 'Shape_Wasserstein' the same way, masked by
#                       M0_MASK_THRESHOLD, for 'density') — see
#                       nn/optimization.py::_FINAL_STEP_SS_WASSERSTEIN_BETA.
#                       Added in v12 so that model/checkpoint SELECTION is
#                       structurally consistent with what's actually being
#                       TRAINED for (the loss now includes an auxiliary
#                       Wasserstein term, wasserstein_loss_weight) — using
#                       plain final_step_SS here would mean picking the best
#                       trial/epoch by a criterion blind to the exact
#                       peak-separation quality the Wasserstein term exists
#                       to improve, which is the thing we actually care about
#                       reporting. The blend weight is a separate, fixed
#                       constant, deliberately NOT the trial's own
#                       wasserstein_loss_weight (see that constant's comment
#                       for why using the tunable per-trial weight here would
#                       corrupt cross-trial comparison).
OBJECTIVE_METRIC = "Hs_SS" if target == "hs" else "final_step_SS_wasserstein"

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
        f"- **Architecture**: (fill in manually)\n"
    )


# Different lead-time studies (12h/24h/48h) all share one
# optuna_study_{STUDY_VERSION}.db file and are commonly launched as separate
# concurrent processes. Plain sqlite:/// gives each writer only Python
# sqlite3's default 5s busy-timeout, which isn't always enough under
# contention -- a collision there surfaces as optuna.exceptions.
# StorageInternalError ("exceeding max length" is just the generic message
# text) during a trial's final state/value commit, crashing the whole run
# and leaving that trial stuck as RUNNING forever. `timeout` makes each
# connection wait out a lock instead of erroring immediately, which is
# enough by itself since contention windows here are just brief per-trial
# commits, not sustained overlapping writes.
#
# Deliberately NOT switching journal_mode to WAL despite WAL being the more
# thorough fix for concurrent writers: the VS Code "Optuna Dashboard"
# extension (right-click a .db file -> Open in Optuna Dashboard) reads the
# file through a single-file sqlite-wasm VFS in the browser sandbox that
# can't follow a WAL database's paired .db-wal/.db-shm side files, and its
# own bundled code (dist/web/storage.worker.js) force-runs
# `pragma journal_mode=DELETE` the instant it opens a file. Point it at a
# WAL-mode db and it reads a stale/incomplete single-file snapshot -- this
# broke the extension for the *entire* study history, not just new trials,
# the first time it was tried.
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
    )

    # Run optuna
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=15, multivariate=True, seed=42
    )
    # 30-step warmup avoids the over-pruning seen at exactly epoch 20 in earlier
    # studies (54% of 12h trials pruned at the boundary with n_warmup_steps=20).
    # interval_steps=5 matches _train_model's PRUNER_SMOOTHING_WINDOW=5 trailing
    # mean, so each check compares fresh, largely non-overlapping windows
    # instead of re-checking the same slow-moving average every single epoch.
    median_pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=30, n_min_trials=5, interval_steps=5
    )
    # v10 analysis of the 24h study (see shape_v10 results) showed trials
    # pruned right at the tf_ratio-decay/warmup boundary with scores
    # statistically indistinguishable from the eventual best trial at that
    # same epoch — the best trial itself dipped and recovered several times
    # before reaching its peak ~30 epochs later. PatientPruner requires
    # `patience` consecutive non-improving reports (on top of MedianPruner's
    # own verdict) before actually pruning, so a trial must be stuck, not just
    # dipping, before it's cut.
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
