"""
Small, fixed-architecture Optuna studies for the KL/Wasserstein/peak
composite-loss ablation. See manuscript/decisions/log/026 for the result
and manuscript/decisions/wasserstein_kl_justification.tex for the loss
terms' own citation-backed justification.

Unlike scripts/optimize.py, this does NOT search architecture/training
hyperparameters — those are PINNED to shape_v12's lead_12h reference config
(results/shape_v12/shape/lead_12h/current_best.txt) so every arm below is a
controlled comparison of LOSS FUNCTION CHOICE alone, holding everything
else fixed. Only Optuna's TPE/pruner machinery is reused, over a 1- or
2-dimensional search space per phase.

IMPORTANT: SpectralWassersteinLoss switched from Wasserstein-1 to
Wasserstein-2 on 2026-08-17 (manuscript/decisions/log/024) — shape_v12's own
wasserstein_loss_weight values (e.g. 181.67 at lead_12h) were tuned under
the OLD W1 metric and are NOT reused here; the 'wasserstein' phase below
retunes it from scratch under W2.

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
    wasserstein_only  base_loss_weight=0, kl_loss_weight=0 (no dependency
                 on 'kl' finishing), searches wasserstein_loss_weight
                 alone substituting the per-bin loss outright. Answers a
                 different question than 'wasserstein' below: does
                 Wasserstein need KL underneath it to be useful, or does
                 it work fine on its own?
    (peak_only   REMOVED 2026-10-02 -- see manuscript/decisions/log/032.
                 It asked wasserstein_only's question for the peak term, and
                 the answer is settled: SoftPeakHeightLoss cannot substitute
                 the per-bin loss, for a structural reason, so re-running it
                 only ever burns GPU confirming it again. The term is one
                 scalar per true peak window (~2.5 per sample against 47
                 outputs), its softmax support is confined to whichever bins
                 the PREDICTION currently peaks at, and it is by design
                 translation-invariant -- so its minimiser is a manifold of
                 spectra, not the true one. The v3 run drove its training
                 loss 5.9 -> 1.9 while validation CC stayed at 0.03-0.08:
                 the optimiser succeeded and the objective did not identify
                 the target. Do NOT re-add this phase.)
    wasserstein  base_loss_weight=0, kl_loss_weight fixed at 'kl' phase's
                 winner, searches wasserstein_loss_weight only (fresh
                 range, see the W1->W2 note above).
    peak         base_loss_weight=0, kl_loss_weight fixed at 'kl' phase's
                 winner, searches peak_loss_weight only. NOTE: this IS the
                 "KL + Peak, no Wasserstein" combination -- its own
                 fixed_loss_weights (saved in best_model.pt) already
                 records kl_loss_weight alongside the searched
                 peak_loss_weight, so no separate phase is needed for that
                 specific question.
    combined     base_loss_weight=0, kl_loss_weight fixed, searches
                 (wasserstein_loss_weight, peak_loss_weight) jointly in a
                 range centered on the 'wasserstein'/'peak' phases'
                 individual winners — L = D_KL + lambda_1*W2 + lambda_2*
                 L_peak, the original proposal, assembled from each term's
                 own best individually-tuned weight as the starting point.

Every phase's search RANGE below is a first guess — widen if a phase's
trials cluster at either edge.

Scoreboard: judge each phase's winner using the wind-sea/swell-conditioned
panel from utils/spectral_peaks.py::peak_modality_metrics (Peak_Height_
RelError_windsea/_swell, Peak_Separation_Recall_windsea/_swell, Tm02_RMSE_
windsea/_swell) plus the whole-spectrum Tm02_RMSE/Bias, NOT Shape_RMSE/
Shape_SS (see manuscript/decisions/log/026). Run scripts/compare_versions.py
or a bespoke evaluate() pass with compute_peak_metrics=True against each
phase's best_model.pt for this.

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
from utils import get_freqs, set_seed, get_device, empty_cache, save_progress, require_slurm

set_seed(42)

BUOY_ID = "32012"
# v1's DB/results (baseline/kl/wasserstein completed, peak killed ~12h in)
# used 'final_step_SS' and are incomparable/superseded — see
# manuscript/decisions/log/009 for why v2 moved to a peak-fidelity objective.
#
# v3 (2026-09-30): v2's results are superseded in turn, for two independent
# reasons, both from decision 030.
#   1. The peak DETECTOR was wrong (criterion 3 measured to the neighbouring
#      maximum rather than the trough; spurious partitions dropped instead of
#      combined). Truth went from 4.21 to 2.53 peaks/spectrum, so every
#      number in results/lossablation_comparison_v2.md scores a panel that no
#      longer exists.
#   2. The same detector backs SoftPeakHeightLoss's training windows
#      (nn/training_loop.py::_peak_windows_for_batch), median 5 -> 16 bins.
#      v2's weights were therefore tuned against a peak LOSS that has since
#      changed shape — not just a changed metric.
# Decision 031 additionally redefined the objective (see OBJECTIVE_METRIC).
# v2's DB and results/ directories are left in place untouched.
STUDY_VERSION = "lossablation_v3"
LEAD_TIME_HOURS = 12
TARGET = "shape"
CHANNEL_SET = "full"
AUX_SET = "dmd"
OBJECTIVE_METRIC = "peak_fidelity"  # applied uniformly to every phase,
                                     # including 'baseline', for a consistent
                                     # cross-phase comparison — see
                                     # manuscript/decisions/log/009, and 031
                                     # for the rename + redefinition (it now
                                     # carries a false-positive term, which
                                     # the recall-only v2 form lacked).
# Effectively inert under the 030 detector: spectra with >4 significant peaks
# fell from 39.5% to 1.3% of the test split, so this cap almost never binds
# now. Left at 4 rather than retuned as a side effect of this re-run.
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

# 'peak_only' is deliberately absent — see the module docstring and
# manuscript/decisions/log/032. This dict is the single source of the phase
# list (compare_ablation_phases.py, evaluate_ablation_phases.py and
# plot_ablation_spectra.py all do PHASES = list(PHASE_N_TRIALS)), so dropping
# it here removes the arm everywhere at once.
PHASE_N_TRIALS = {"baseline": 5, "kl": 15, "wasserstein_only": 15,
                  "wasserstein": 15, "peak": 15, "combined": 18}
PHASE_N_STARTUP = {"baseline": 5, "kl": 5, "wasserstein_only": 5,
                   "wasserstein": 5, "peak": 5, "combined": 8}


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
    if phase == "wasserstein_only":
        return 0.0, dict(kl_loss_weight=0.0, peak_loss_weight=0.0)
    if phase == "peak_only":
        # Removed deliberately, not an oversight — see the module docstring
        # and manuscript/decisions/log/032. Named explicitly so re-adding it
        # is a conscious act rather than something a stale --phase argument
        # resurrects silently.
        raise ValueError(
            "Phase 'peak_only' was removed (decision 032): SoftPeakHeightLoss is "
            "structurally improper as a standalone objective — ~2.5 scalar constraints "
            "on 47 outputs, support confined to the prediction's own current maxima, and "
            "translation-invariant by design, so its minimiser is a manifold rather than "
            "the true spectrum. The v3 run confirmed it (train loss 5.9->1.9, val CC "
            "0.03-0.08). Use 'peak' (KL + peak) instead."
        )
    # Reject an unknown phase BEFORE the disk read below. The read is what
    # makes this ordering matter: with it first, a typo'd phase name
    # surfaced as FileNotFoundError("run --phase kl first") — which is both
    # the wrong error and actively misleading advice. It only ever looked
    # correct because a completed 'kl' happened to be on disk for the
    # current STUDY_VERSION; bumping the version made every unknown name
    # fail that way instead.
    if phase not in ("wasserstein", "peak", "combined"):
        raise ValueError(f"Unknown phase {phase!r}")
    # Everything below builds on 'kl's winning weight -- no dependency on it
    # for wasserstein_only above, which is exactly its point: does
    # Wasserstein need KL underneath to substitute the per-bin loss (as
    # 'wasserstein'/'peak' below assume), or does it work fine alone? That
    # question is only worth asking for a term that COULD stand alone; W2 is
    # a full-support transport distance, so it can. The peak term is not
    # (decision 032), which is why there is no 'peak_only' counterpart.
    kl_w = _read_prior_weight("kl", "kl_loss_weight")
    if phase == "wasserstein":
        return 0.0, dict(kl_loss_weight=kl_w, peak_loss_weight=0.0)
    if phase == "peak":
        return 0.0, dict(kl_loss_weight=kl_w, wasserstein_loss_weight=0.0)
    return 0.0, dict(kl_loss_weight=kl_w)  # 'combined'


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
            # v3: upper bound 50 -> 200. v2's winner was 45.87, i.e. hard
            # against the old ceiling, which is exactly the "widen if a
            # phase's trials cluster at either edge" case the module
            # docstring calls for.
            weights["kl_loss_weight"] = trial.suggest_float("kl_loss_weight", 0.1, 200.0, log=True)
        elif phase in ("wasserstein", "wasserstein_only"):
            # Same range for both -- same parameter, same mechanism; the
            # only difference is whether kl_loss_weight is fixed nonzero
            # (see _fixed_weights_for_phase) or pinned to 0 alongside it.
            # v3: 200 -> 1000, v2's 'wasserstein' winner was 154.0 (near the
            # old ceiling), though 'wasserstein_only' settled at 9.5 -- the
            # wider bracket covers both without forcing a choice.
            weights["wasserstein_loss_weight"] = trial.suggest_float(
                "wasserstein_loss_weight", 1.0, 1000.0, log=True)
        elif phase == "peak":
            # v3: [0.01, 20] -> [0.001, 100]. v2's 'peak_only' winner sat at
            # 0.0108, essentially ON the old floor. The bracket also has to
            # move because SoftPeakHeightLoss's windows widened (median 5 ->
            # 16 bins under the 030 detector), so the term's magnitude per
            # unit weight is not what it was when [0.01, 20] was chosen.
            # Note the upper end is only safe BECAUSE kl_loss_weight is fixed
            # nonzero in this phase: the peak term carries no position
            # information of its own, so letting it dominate the gradient
            # walks back toward the removed 'peak_only' regime (decision 032).
            weights["peak_loss_weight"] = trial.suggest_float(
                "peak_loss_weight", 0.001, 100.0, log=True)
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
                compute_peak_metrics=True,
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
                    # PINNED_CONFIG first so trial.params wins on any overlap.
                    # trial.params alone holds ONLY this phase's searched loss
                    # weight -- the architecture is pinned, so it never passes
                    # through trial.suggest_* and never lands there. That made
                    # these checkpoints unloadable by anything built on
                    # nn.checkpoints.build_model (scripts/infer.py,
                    # compare_versions.py, compare_physical_baseline.py), which
                    # reads embed_dim as params['head_dim'] * params['nhead'];
                    # scripts/evaluate_ablation_phases.py only works because it
                    # re-imports PINNED_CONFIG itself. Same failure mode as the
                    # fixed_head_dim gap in nn/optimization.py (decision 031):
                    # 'params' must be a complete reconstruction recipe.
                    'params': {**PINNED_CONFIG, **trial.params},
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
            # v3 adds Peak_Separation_Precision (new in decision 031), the
            # peak-count pair, and Tm02_Bias_*. The bias keys were the gap
            # that forced scripts/evaluate_ablation_phases.py to re-run a
            # full test pass just to recover them for
            # compare_ablation_phases.py's '[val]'-sourced rows.
            for key in ['RMSE', 'Hs_MAPE', 'CC', 'Bias', 'R2', 'overall_SS',
                        'Shape_RMSE', 'Shape_SS', 'Shape_Mass_Error',
                        'Peak_Height_RelError_windsea', 'Peak_Height_RelError_swell',
                        'Peak_Separation_Recall_windsea', 'Peak_Separation_Recall_swell',
                        'Peak_Separation_Precision',
                        'Peak_Count_True_Mean', 'Peak_Count_Pred_Mean',
                        'Peak_windsea_n', 'Peak_swell_n',
                        'Tm02_RMSE_windsea', 'Tm02_RMSE_swell',
                        'Tm02_Bias_windsea', 'Tm02_Bias_swell']:
                if key in best_val_metrics:
                    trial.set_user_attr(f'val_{key}', best_val_metrics[key])
            # Training-loss composition at the selected epoch — the per-phase
            # analogue of what nn/optimization.py::objective records. For this
            # study it is the direct read on whether an arm ran balanced or
            # dominated by a single term (decisions 032, 033).
            for key in ('base', 'w2', 'kl', 'peak'):
                share_key = f'train_loss_share_{key}'
                if share_key in best_val_metrics:
                    trial.set_user_attr(share_key, best_val_metrics[share_key])

        return best_val_score

    return objective


def main():
    require_slurm("scripts/ablate_loss.py")

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", required=True,
                         choices=["baseline", "kl", "wasserstein_only",
                                  "wasserstein", "peak", "combined"])  # no 'peak_only': decision 032
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
