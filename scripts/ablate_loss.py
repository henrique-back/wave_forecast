"""
Small, fixed-architecture Optuna studies for the KL/Wasserstein/peak
composite-loss ablation (see CLAUDE.md's "Shape/magnitude model split" area
and the loss-ablation discussion around utils/loss.py's SpectralWasserstein
Loss, SpectralKLDivergenceLoss, SoftPeakHeightLoss).

Unlike scripts/optimize.py, this does NOT search architecture/training
hyperparameters — those are PINNED to shape_v12's lead_12h reference config
(results/shape_v12/shape/lead_12h/current_best.txt, 54/70 trials complete
at time of writing) so every arm below is a controlled comparison of LOSS
FUNCTION CHOICE alone, holding everything else fixed. This is deliberate,
not a shortcut: re-searching a ~10-dimensional architecture space with a
~15-trial budget would rediscover nothing reliable (shape_v9/v10/v11's own
70-trial searches never converged to one stable seq_len/num_layers/
batch_size across lead times — see git history / RESEARCH_LOG.md), AND
would confound "did the loss change help" with "did we get luckier on
architecture this run". Only Optuna's TPE/pruner machinery is reused, over
a 1- or 2-dimensional search space per phase.

IMPORTANT: SpectralWassersteinLoss switched from Wasserstein-1 to
Wasserstein-2 on 2026-08-17 (quantile-domain formula, see utils/loss.py's
docstring) — shape_v12's own wasserstein_loss_weight values (e.g. 181.67 at
lead_12h) were tuned under the OLD W1 metric and are NOT reused here; the
'wasserstein' phase below retunes it from scratch under W2.

Phases (run via --phase; each phase after 'kl' depends on the 'kl' phase's
winning kl_loss_weight, read back from its own current_best.txt — run them
in order):
    baseline     base_loss_weight=1, every auxiliary weight=0 — the current
                 production per-bin loss, retrained at the pinned
                 architecture. Run 5x (PHASE_N_TRIALS) with different data
                 shuffles rather than searching anything, to get a
                 run-to-run variance estimate for the scoreboard (same
                 architecture/loss every time — Optuna is just used here as
                 a convenient repeated-runs harness, not a search).
    kl           base_loss_weight=0 (literal SUBSTITUTE of the per-bin loss,
                 matching the original L = D_KL + lambda_1*W2 + lambda_2*
                 L_peak formula, which has no per-bin MSE term at all —
                 see nn/training_loop.py::train_one_epoch's docstring).
                 Searches kl_loss_weight only.
    wasserstein  base_loss_weight=0, kl_loss_weight fixed at 'kl' phase's
                 winner, searches wasserstein_loss_weight only (fresh
                 range, see the W1->W2 note above).
    peak         base_loss_weight=0, kl_loss_weight fixed at 'kl' phase's
                 winner, searches peak_loss_weight only.
    combined     base_loss_weight=0, kl_loss_weight fixed, searches
                 (wasserstein_loss_weight, peak_loss_weight) jointly in a
                 range centered on the 'wasserstein'/'peak' phases'
                 individual winners — L = D_KL + lambda_1*W2 + lambda_2*
                 L_peak, the original proposal, assembled from each term's
                 own best individually-tuned weight as the starting point.

Every phase's search RANGE below is a first guess (no manual pre-sweep
exists yet for kl_loss_weight/peak_loss_weight, unlike wasserstein_loss_
weight's original W1-era 10-400 bracket, itself now stale) — widen if a
phase's trials cluster at either edge, same convention nn/optimization.py's
objective() already uses for its own under-explored ranges (see e.g. lr's
docstring there).

Scoreboard: judge each phase's winner using the wind-sea/swell-conditioned
panel from utils/spectral_peaks.py::peak_modality_metrics (Peak_Height_
RelError_windsea/_swell, Peak_Separation_Recall_windsea/_swell, Tm02_RMSE_
windsea/_swell) plus the whole-spectrum Tm02_RMSE/Bias — NOT Shape_RMSE/
Shape_SS as the primary criterion (see the loss-ablation discussion:
RMSE-family metrics are structurally biased toward the blurry/hedged
predictions these new loss terms exist to move away from). Run
scripts/compare_versions.py or a bespoke evaluate() pass with
compute_peak_metrics=True against each phase's best_model.pt for this.

Usage:
    python scripts/ablate_loss.py --phase baseline
    python scripts/ablate_loss.py --phase kl
    python scripts/ablate_loss.py --phase wasserstein
    python scripts/ablate_loss.py --phase peak
    python scripts/ablate_loss.py --phase combined
"""

import argparse
import ast
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import optuna
import pandas as pd
import torch

from nn import WaveHeightBaselineNN
from nn.optimization import _prepare_dataloaders, _train_model
from utils import get_freqs, set_seed, get_device, empty_cache, save_progress

set_seed(42)

BUOY_ID = "32012"
STUDY_VERSION = "lossablation_v1"
LEAD_TIME_HOURS = 12
TARGET = "shape"
CHANNEL_SET = "full"
AUX_SET = "dmd"
OBJECTIVE_METRIC = "final_step_SS"  # deliberately NOT 'final_step_SS_wasserstein'
                                     # — that blend rewards low Shape_Wasserstein
                                     # specifically, which would bias checkpoint
                                     # selection in favor of whichever arm's own
                                     # loss term happens to move Shape_Wasserstein,
                                     # even for arms (baseline/kl) that never
                                     # touch it. Plain final_step_SS applies the
                                     # same neutral criterion to every phase.
MAX_PEAKS = 4

# Architecture + training hyperparameters pinned from
# results/shape_v12/shape/lead_12h/current_best.txt (54/70 trials complete)
# — wasserstein_loss_weight excluded deliberately (see module docstring).
PINNED_CONFIG = dict(
    seq_len=12,
    batch_size=64,
    lr=0.009801709153151667,
    freq_embed_dropout=0.26604351832485784,
    embed_dropout=0.20554613662356902,
    head_dim=8,
    nhead=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    weight_decay=0.0004182586391136781,
)

PHASE_N_TRIALS = {"baseline": 5, "kl": 15, "wasserstein": 15, "peak": 15, "combined": 18}
PHASE_N_STARTUP = {"baseline": 5, "kl": 5, "wasserstein": 5, "peak": 5, "combined": 8}


def _phase_dir(phase):
    return (Path(__file__).parent.parent / "results" / f"lossablation_{phase}_{STUDY_VERSION}"
            / TARGET / f"lead_{LEAD_TIME_HOURS}h")


def _read_prior_weight(phase, key):
    """Read a previous phase's winning weight back out of its
    current_best.txt (written by utils.save_progress) — mirrors how
    scripts/train.py reads best_trial.txt's params for a final retrain.
    ast.literal_eval, not eval(): the file is locally-written/trusted, but
    literal_eval is the correct tool for parsing a dict literal regardless.
    """
    path = _phase_dir(phase) / "current_best.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/ablate_loss.py --phase {phase}` "
            f"first; this phase's search depends on {phase}'s winning weight."
        )
    line = next(l for l in path.read_text().splitlines() if l.startswith("Best params:"))
    params = ast.literal_eval(line.split("Best params:", 1)[1].strip())
    return params[key]


def _fixed_weights_for_phase(phase):
    """(base_loss_weight, dict of weights NOT searched by this phase —
    either permanently 0 or carried over from a prior phase's winner)."""
    if phase == "baseline":
        return 1.0, dict(kl_loss_weight=0.0, wasserstein_loss_weight=0.0, peak_loss_weight=0.0)
    if phase == "kl":
        return 0.0, dict(wasserstein_loss_weight=0.0, peak_loss_weight=0.0)
    kl_w = _read_prior_weight("kl", "kl_loss_weight")
    if phase == "wasserstein":
        return 0.0, dict(kl_loss_weight=kl_w, peak_loss_weight=0.0)
    if phase == "peak":
        return 0.0, dict(kl_loss_weight=kl_w, wasserstein_loss_weight=0.0)
    if phase == "combined":
        return 0.0, dict(kl_loss_weight=kl_w)
    raise ValueError(f"Unknown phase {phase!r}")


def make_objective(phase, density, alpha_1, alpha_2, r_1, r_2, wind, freqs, results_folder):
    base_loss_weight, fixed = _fixed_weights_for_phase(phase)

    def objective(trial):
        weights = dict(fixed)
        if phase == "kl":
            # No prior manual sweep — see module docstring. base_loss_weight=0
            # here means this weight sets the LOSS'S OVERALL SCALE (there's no
            # base MSE term to balance against), acting more like an effective
            # LR multiplier than a delicate two-term mixing ratio — gradient
            # clipping (max_norm=1.0, see train_one_epoch) bounds the downside
            # of a too-large draw, so a wide bracket is reasonable to explore.
            weights["kl_loss_weight"] = trial.suggest_float("kl_loss_weight", 0.1, 50.0, log=True)
        elif phase == "wasserstein":
            weights["wasserstein_loss_weight"] = trial.suggest_float(
                "wasserstein_loss_weight", 1.0, 200.0, log=True)
        elif phase == "peak":
            weights["peak_loss_weight"] = trial.suggest_float(
                "peak_loss_weight", 0.01, 20.0, log=True)
        elif phase == "combined":
            w_center = _read_prior_weight("wasserstein", "wasserstein_loss_weight")
            p_center = _read_prior_weight("peak", "peak_loss_weight")
            weights["wasserstein_loss_weight"] = trial.suggest_float(
                "wasserstein_loss_weight", w_center / 3.0, w_center * 3.0, log=True)
            weights["peak_loss_weight"] = trial.suggest_float(
                "peak_loss_weight", p_center / 3.0, p_center * 3.0, log=True)
        # 'baseline': nothing to sample — every weight is fixed at 0 (see
        # _fixed_weights_for_phase); trial.number still seeds the data
        # shuffle below, so these 5 runs are variance replicates, not a
        # search.

        device = get_device()
        (train_loader, val_loader, test_loader, freq_means, shape_means,
         num_freqs, num_channels, num_aux_channels) = _prepare_dataloaders(
            density, alpha_1, alpha_2, r_1, r_2,
            PINNED_CONFIG["seq_len"], LEAD_TIME_HOURS, PINNED_CONFIG["batch_size"], TARGET,
            shuffle_seed=trial.number, wind=wind, channel_set=CHANNEL_SET, aux_set=AUX_SET,
        )

        embed_dim = PINNED_CONFIG["head_dim"] * PINNED_CONFIG["nhead"]
        model = WaveHeightBaselineNN(
            num_freqs=num_freqs,
            freqs=freqs,
            target=TARGET,
            num_channels=num_channels,
            num_aux_channels=num_aux_channels,
            freq_embed_dropout=PINNED_CONFIG["freq_embed_dropout"],
            embed_dropout=PINNED_CONFIG["embed_dropout"],
            nhead=PINNED_CONFIG["nhead"],
            num_encoder_layers=PINNED_CONFIG["num_encoder_layers"],
            num_decoder_layers=PINNED_CONFIG["num_decoder_layers"],
            embed_dim=embed_dim,
        ).to(device)

        try:
            best_val_score, best_val_metrics, best_model_state = _train_model(
                model, train_loader, val_loader, device, freqs, freq_means, shape_means,
                TARGET, LEAD_TIME_HOURS, PINNED_CONFIG["lr"], PINNED_CONFIG["weight_decay"],
                OBJECTIVE_METRIC, num_epochs=100, patience=20, trial=trial,
                base_loss_weight=base_loss_weight,
                kl_loss_weight=weights.get("kl_loss_weight", 0.0),
                wasserstein_loss_weight=weights.get("wasserstein_loss_weight", 0.0),
                peak_loss_weight=weights.get("peak_loss_weight", 0.0),
                peak_max_count=MAX_PEAKS,
            )
        except torch.OutOfMemoryError:
            del model
            empty_cache(device)
            raise

        # Checkpoint whenever this trial beats every trial completed so far —
        # same convention as nn/optimization.py::objective().
        if results_folder is not None and best_model_state is not None:
            try:
                current_best = trial.study.best_value
            except ValueError:
                current_best = float('-inf')
            if best_val_score > current_best:
                torch.save({
                    'model_state_dict': best_model_state,
                    'params': trial.params,
                    'target': TARGET,
                    'lead_time_steps': LEAD_TIME_HOURS,
                    'freq_means': freq_means,
                    'shape_means': shape_means,
                    'freqs': freqs,
                    'trial_number': trial.number,
                    'val_score': best_val_score,
                    # Ablation-specific: the full loss recipe this checkpoint
                    # was trained with, since 'params' only has the SEARCHED
                    # weight(s) — needed to reconstruct/report the composite
                    # loss later without re-deriving it from the phase name.
                    'base_loss_weight': base_loss_weight,
                    'fixed_loss_weights': fixed,
                }, Path(results_folder) / 'best_model.pt')

        if best_val_metrics is not None:
            for key in ['RMSE', 'Hs_MAPE', 'CC', 'Bias', 'R2', 'overall_SS',
                        'Shape_RMSE', 'Shape_SS', 'Shape_Mass_Error']:
                if key in best_val_metrics:
                    trial.set_user_attr(f'val_{key}', best_val_metrics[key])

        return best_val_score

    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True,
                         choices=["baseline", "kl", "wasserstein", "peak", "combined"])
    args = parser.parse_args()
    phase = args.phase

    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "buoy_data" / BUOY_ID / "processed_data.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found — run scripts/data_processing.py first.")
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(file_path)
    freqs = get_freqs(density)

    results_folder = _phase_dir(phase)
    results_folder.mkdir(parents=True, exist_ok=True)

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///optuna_study_{STUDY_VERSION}.db",
        engine_kwargs={"connect_args": {"timeout": 30}},
    )
    study_name = f"{TARGET}_{CHANNEL_SET}_{AUX_SET}_lead_{LEAD_TIME_HOURS}h_{phase}_{STUDY_VERSION}"

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=PHASE_N_STARTUP[phase], multivariate=True, seed=42
    )
    median_pruner = optuna.pruners.MedianPruner(n_warmup_steps=30, n_min_trials=5, interval_steps=5)
    pruner = optuna.pruners.PatientPruner(median_pruner, patience=10, min_delta=0.0)

    study = optuna.create_study(
        study_name=study_name, storage=storage, direction="maximize",
        load_if_exists=True, sampler=sampler, pruner=pruner,
    )

    objective_fn = make_objective(phase, density, alpha_1, alpha_2, r_1, r_2, wind, freqs, results_folder)
    study.optimize(
        objective_fn,
        n_trials=PHASE_N_TRIALS[phase],
        callbacks=[lambda study, trial: save_progress(study, trial, results_folder)],
        catch=(torch.OutOfMemoryError,),
    )

    print(f"\n=== Phase {phase!r} done ===")
    print("Best trial params:", study.best_trial.params)
    print("Best value:", study.best_value)

    with open(results_folder / "best_trial.txt", "w") as f:
        f.write(f"Phase: {phase}\nLead time (hours): {LEAD_TIME_HOURS}\n")
        f.write(f"Best trial parameters:\n{study.best_trial.params}\n")
        for key, val in study.best_trial.user_attrs.items():
            f.write(f"{key}: {val}\n")


if __name__ == "__main__":
    main()
