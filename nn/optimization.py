import math
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from nn import (WaveSpectralDataset, WaveHeightBaselineNN, prepare_X, prepare_aux, prepare_y,
                 train_one_epoch, evaluate, compute_dmd_features)
from nn.channels import CHANNEL_SETS, NORM_MODES, AUX_CHANNEL_SETS, AUX_NORM_MODES
from utils import set_seed, get_device, empty_cache


def _seed_worker(worker_id):
    torch.manual_seed(42 + worker_id)


def _normalize(train_df, *other_dfs, mode='zscore'):
    """Fit normalization on train_df and apply to all DataFrames.

    mode='zscore': subtract mean, divide by std.  May produce negative values —
        use for channels that are not fed into physical computations (alpha, r1).
    mode='scale':  divide by per-column mean only.  Preserves non-negativity —
        required for spectral density, which is passed to compute_hs / sqrt().
    mode='none':   pass through unchanged — for channels already on a fixed,
        meaningful scale (e.g. sin/cos of a circular angle, already in [-1, 1]).
    """
    if mode == 'zscore':
        mean = train_df.mean()
        std = train_df.std().clip(lower=1e-8)
        return tuple((df - mean) / std for df in (train_df, *other_dfs))
    elif mode == 'scale':
        mean = train_df.mean().clip(lower=1e-8)
        return tuple(df / mean for df in (train_df, *other_dfs))
    else:  # 'none'
        return (train_df, *other_dfs)


# Fixed blend weight for the 'final_step_SS_wasserstein' objective_metric —
# deliberately not the trial's own tunable wasserstein_loss_weight, which
# would make cross-trial comparison unfair. Order-of-magnitude estimate, not
# precisely fit — see manuscript/decisions/log/021.
_FINAL_STEP_SS_WASSERSTEIN_BETA = 10.0

# Median raw magnitude of each term nn/training_loop.py::train_one_epoch sums,
# measured by scripts/measure_loss_term_scales.py on the lossablation_v3
# 'kl'-phase checkpoint (buoy 32012, target='shape', lead 12h, post-030
# detector, 6 training batches under teacher forcing).
#
# These exist because the four terms are summed RAW while living in four
# unrelated units — log-space squared error, Hz, nats, squared physical E² —
# spanning ~4.5 orders of magnitude. A raw weight therefore says nothing about
# how much its term actually contributes, which is how v13's independently
# sampled weights produced a space whose MEDIAN draw was ~350:1 peak-dominated
# (decision 033). objective() samples a CONTRIBUTION per term and divides by
# the reference here, so a sampled value means "this term is worth this much
# of the loss".
#
# Re-derive (and record why in the decision log) whenever something changes
# what a term measures: the peak detector — decision 030's combining moved the
# median window 5 -> 16 bins, which rescales L_peak — the target, the
# normalisation, or the frequency grid. A stale reference silently un-centres
# the search space; measure_loss_term_scales.py flags drift beyond 2x.
LOSS_TERM_REFERENCE = {'base': 6.84, 'kl': 0.064, 'w2': 0.0085, 'peak': 197.0}


def resolve_loss_weights(params: dict) -> dict:
    """Absolute {kl,wasserstein,peak}_loss_weight from a trial's params dict.

    v14+ studies sample a contribution and two ratios relative to KL
    ('kl_contrib', 'w2_rel', 'peak_rel' — see objective()); v13 and earlier
    sampled the three absolute weights directly. Both forms appear in
    best_trial.txt/current_best.txt across the results tree, so scripts/train.py
    reads them through here rather than indexing the raw keys — a v13 retrain
    has to keep working.

    Raises KeyError if params carries neither form. That is deliberate: the
    caller previously used params.get(key, 0.0), so a renamed parameter
    silently produced an all-zero loss and surfaced much later as a confusing
    "this best_trial.txt predates the KL/Wasserstein/Peak loss" error.
    """
    if 'kl_contrib' in params:          # v14+
        kl_contrib = params['kl_contrib']
        return {
            'kl_loss_weight': kl_contrib / LOSS_TERM_REFERENCE['kl'],
            'wasserstein_loss_weight': (kl_contrib * params['w2_rel']
                                         / LOSS_TERM_REFERENCE['w2']),
            'peak_loss_weight': (kl_contrib * params['peak_rel']
                                  / LOSS_TERM_REFERENCE['peak']),
        }
    if 'kl_loss_weight' in params:      # v13 and earlier
        return {k: params.get(k, 0.0) for k in
                ('kl_loss_weight', 'wasserstein_loss_weight', 'peak_loss_weight')}
    raise KeyError(
        "params carries neither the v14+ loss parameterisation ('kl_contrib', "
        "'w2_rel', 'peak_rel') nor the pre-v14 absolute weights "
        f"('kl_loss_weight', ...). Got keys: {sorted(params)}")


def _weighted_mean_ss(per_step_ss):
    """Exponentially-weighted mean Skill Score.

    Later forecast steps are downweighted so that strongly-negative SS at long
    horizons does not mask genuine improvements at short horizons.  The weight
    halves at the midpoint of the forecast horizon, reaching 0.25 at the last
    step, so late steps still contribute but cannot dominate.
    """
    n = len(per_step_ss)
    half_life = max(1.0, n / 2.0)
    weights = [math.exp(-t * math.log(2) / half_life) for t in range(n)]
    weight_sum = sum(weights)
    return sum(w * ss for w, ss in zip(weights, per_step_ss)) / weight_sum


def _compute_val_score(metrics: dict, objective_metric: str) -> float:
    """Return a 'higher is better' scalar for the given metric name.

    Skill Scores are already higher-is-better; error metrics (RMSE,
    Hs_RMSE, etc.) are negated. Valid values for objective_metric:
    'final_step_SS', 'weighted_mean_SS', 'overall_SS', 'Hs_SS', 'RMSE',
    'Hs_RMSE', 'Tm02_RMSE', 'Shape_RMSE', 'SI_mean',
    'final_step_SS_wasserstein', 'peak_fidelity'.

    'peak_fidelity' is the odd one out: NOT a transform of RMSE, unlike every
    other option. It is built from utils.spectral_peaks.peak_modality_metrics
    (target=='shape' only; requires compute_peak_metrics=True on the
    evaluate() call, KeyError otherwise — deliberately not falling back to an
    RMSE-rooted metric; float('-inf') if no true peak was detected in either
    label):

        R  = nanmean(Peak_Separation_Recall_windsea, _swell)   macro, [0, 1]
        P  = Peak_Separation_Precision                         pooled, [0, 1]
        F1 = 2PR / (P + R)
        H  = 1 - min(nanmean(Peak_Height_RelError_windsea, _swell), 1)
        peak_fidelity = F1 * H                                         [0, 1]

    Needed because scripts/ablate_loss.py trains several arms on a loss that
    isn't RMSE at all (base_loss_weight=0), so an RMSE-rooted selection
    metric would bias trial/epoch selection back toward RMSE-friendly
    behaviour regardless of whether peak fidelity actually improved.

    Recall stays a macro average over the two partition labels so a model
    cannot ignore whichever regime is rarer; precision is pooled, since a
    predicted peak matching nothing has no true partition to take a label
    from (see peak_modality_metrics).

    Renamed from 'peak_fidelity_SS' and redefined on 2026-09-30: the old form
    was recall - rel_err, which (a) had no false-positive term, so it was
    maximised by over-segmenting, and (b) subtracted an unbounded ratio from
    a bounded fraction. Both terms are now bounded and combined
    multiplicatively, so neither can be traded away. It is not, and never
    was, a skill score — no baseline appears in it — hence dropping the _SS.
    Old and new numbers are NOT comparable; the name change is what makes a
    stale caller fail loudly here rather than silently rescale. See
    manuscript/decisions/log/ 001 ('weighted_mean_SS'), 009/026/031
    ('peak_fidelity'), 014 ('final_step_SS'), 021
    ('final_step_SS_wasserstein'), 030 (the detector fix that exposed this).
    """
    if objective_metric == 'final_step_SS':
        return metrics['per_step_SS'][-1]
    elif objective_metric == 'weighted_mean_SS':
        return _weighted_mean_ss(metrics['per_step_SS'])
    elif objective_metric == 'overall_SS':
        return metrics['overall_SS']
    elif objective_metric == 'Hs_SS':
        return metrics['Hs_SS']
    elif objective_metric == 'RMSE':
        return -metrics['RMSE']
    elif objective_metric == 'Hs_RMSE':
        return -metrics['Hs_RMSE']
    elif objective_metric == 'Tm02_RMSE':
        return -metrics['Tm02_RMSE']
    elif objective_metric == 'Shape_RMSE':
        return -metrics['Shape_RMSE']
    elif objective_metric == 'SI_mean':
        return -metrics['SI_mean']
    elif objective_metric == 'final_step_SS_wasserstein':
        return metrics['per_step_SS'][-1] - _FINAL_STEP_SS_WASSERSTEIN_BETA * metrics['Shape_Wasserstein']
    elif objective_metric == 'peak_fidelity':
        # nanmean over an all-NaN slice raises numpy's "Mean of empty slice"
        # RuntimeWarning (Python's warnings machinery, not an IEEE-754
        # errstate one -- np.errstate doesn't touch it) -- checked for and
        # handled explicitly below, so filter it here rather than let it
        # spam the log on every epoch of a run with genuinely no detectable
        # peaks in one label.
        rel_errs = [metrics['Peak_Height_RelError_windsea'], metrics['Peak_Height_RelError_swell']]
        recalls = [metrics['Peak_Separation_Recall_windsea'], metrics['Peak_Separation_Recall_swell']]
        precision = metrics['Peak_Separation_Precision']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            rel_err = float(np.nanmean(rel_errs))
            recall = float(np.nanmean(recalls))
        if np.isnan(rel_err) or np.isnan(recall):
            return float('-inf')
        # A pass that predicted no peaks at all leaves precision undefined;
        # that is zero peak fidelity, not missing data, so it scores 0 rather
        # than dropping out of the harmonic mean and letting recall stand
        # alone (recall would be 0 there anyway, but be explicit).
        if np.isnan(precision):
            precision = 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        # Height agreement, bounded into [0, 1] the same way: rel_err is a
        # ratio with no upper limit, so the pre-2026-09-30 form (recall -
        # rel_err) let a single badly-missed peak height dominate a score
        # whose other term was a bounded fraction, at an implicit 1:1 weight
        # that was never justified. Both factors are now fractions and the
        # product is in [0, 1], 1 being a perfect forecast.
        height_agreement = 1.0 - min(rel_err, 1.0)
        return f1 * height_agreement
    else:
        raise ValueError(
            f"Unknown objective_metric {objective_metric!r}. Valid: "
            "'final_step_SS', 'weighted_mean_SS', 'overall_SS', 'Hs_SS', "
            "'RMSE', 'Hs_RMSE', 'Tm02_RMSE', 'Shape_RMSE', 'SI_mean', "
            "'final_step_SS_wasserstein', 'peak_fidelity'"
        )


def _train_model(model, train_loader, val_loader, device, freqs, freq_means,
                  shape_means, target, lead_time, lr, weight_decay, objective_metric,
                  num_epochs=80, patience=10, trial=None, wasserstein_loss_weight=0.0,
                  kl_loss_weight=0.0, base_loss_weight=1.0, peak_loss_weight=0.0,
                  peak_max_count=4, compute_peak_metrics=False):
    """Run the scheduled-sampling training loop with early stopping.

    Shared by objective() (Optuna trial) and scripts/train.py (fixed-config
    final retrain) so the two never drift apart. When `trial` is given,
    reports the per-epoch score to Optuna and prunes on its signal; this is
    the only behavioural difference between the two callers.

    compute_peak_metrics : bool, default False — forwarded to every per-epoch
        evaluate(...) call. Only needs to be True for objective_metric ==
        'peak_fidelity' (scripts/ablate_loss.py).

    wasserstein_loss_weight, kl_loss_weight, base_loss_weight,
    peak_loss_weight, peak_max_count : forwarded to train_one_epoch's
        auxiliary loss terms — see that function's docstring for what each
        one does. See manuscript/decisions/log/020 (Wasserstein term),
        024 (its W1->W2 switch), 026 (the composite-loss ablation that
        validated kl_loss_weight/peak_loss_weight).

    Returns (best_val_score, best_val_metrics, best_model_state) — note
    best_val_score is the SMOOTHED score (see VAL_SCORE_SMOOTHING_WINDOW
    below), not a single epoch's raw value.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # patience=3 so the LR is halved 7 epochs before early stopping fires (at
    # patience=10), giving the model meaningful time to benefit from the new LR.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, cooldown=2
    )

    # Ramps AdamW up to the sampled lr over the first few epochs instead of
    # applying it from epoch 0. ReduceLROnPlateau only starts stepping once
    # warmup ends. See manuscript/decisions/log/022.
    WARMUP_EPOCHS = 5

    # tf_ratio decays linearly from 1.0 to 0.0 over tf_decay_epochs.
    tf_decay_epochs = 2 * patience

    # Trailing mean smoothing every val_score-driven decision (LR scheduler,
    # early-stopping/checkpoint selection, and the Optuna pruner report) from
    # the same window, not just the pruner as originally — see
    # manuscript/decisions/log/023.
    VAL_SCORE_SMOOTHING_WINDOW = 5
    val_score_history = []

    best_val_score = float('-inf')
    best_val_metrics = None
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        if epoch < WARMUP_EPOCHS:
            # 10% -> 100% of the sampled lr over WARMUP_EPOCHS epochs, rather
            # than 0% -> 100%, so the very first step still makes progress.
            warmup_lr = lr * (0.1 + 0.9 * (epoch + 1) / WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        tf_ratio = max(0.0, 1.0 - epoch / tf_decay_epochs)

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, freqs,
                                        tf_ratio=tf_ratio, freq_means=freq_means,
                                        shape_means=shape_means,
                                        wasserstein_loss_weight=wasserstein_loss_weight,
                                        kl_loss_weight=kl_loss_weight,
                                        base_loss_weight=base_loss_weight,
                                        peak_loss_weight=peak_loss_weight,
                                        peak_max_count=peak_max_count)
        val_metrics   = evaluate(model, val_loader, device, freqs,
                                  lead_time=lead_time, freq_means=freq_means,
                                  shape_means=shape_means,
                                  compute_peak_metrics=compute_peak_metrics)

        val_score = _compute_val_score(val_metrics, objective_metric)
        val_score_history.append(val_score)
        smoothed_score = float(np.mean(val_score_history[-VAL_SCORE_SMOOTHING_WINDOW:]))

        if epoch >= WARMUP_EPOCHS:
            scheduler.step(smoothed_score)

        bulk_str = ""
        if target == 'density' and 'Hs_RMSE' in val_metrics:
            bulk_str = (f" | Val Hs_RMSE: {val_metrics['Hs_RMSE']:.4f}"
                        f" | Val Hs_Bias: {val_metrics['Hs_Bias']:+.4f}"
                        f" | Val Tm02_RMSE: {val_metrics['Tm02_RMSE']:.4f}"
                        f" | Val Tm02_Bias: {val_metrics['Tm02_Bias']:+.4f}"
                        f" | Val Shape_RMSE: {val_metrics['Shape_RMSE']:.4f}"
                        f" (masked: {val_metrics['Shape_masked_samples']})"
                        f" | Val Shape_SS: {val_metrics['Shape_SS']:.4f}"
                        f" | Val SI_mean: {val_metrics['SI_mean']:.4f}")
        elif target == 'shape' and 'Shape_RMSE' in val_metrics:
            bulk_str = (f" | Val Shape_RMSE: {val_metrics['Shape_RMSE']:.4f}"
                        f" | Val Shape_SS: {val_metrics['Shape_SS']:.4f}"
                        f" | Val Shape_Mass_Error: {val_metrics['Shape_Mass_Error']:.6f}")
        # Hs_MAPE is only populated for 'hs'/'density' targets (see evaluate.py);
        # 'shape' has no magnitude to compute a MAPE against.
        hs_mape_str = (f"{val_metrics['Hs_MAPE']:.2f}%"
                       if val_metrics['Hs_MAPE'] is not None else "N/A")
        # Per-term shares of the training loss, printed only when a composite
        # is actually in play (more than one term carries weight). A run
        # dominated by one term should be obvious from epoch 1 rather than
        # after a whole study — see decisions 032/033.
        components = train_metrics.get('loss_components') or {}
        active = {k: v for k, v in components.items() if v != 0.0}
        comp_total = sum(abs(v) for v in active.values())
        loss_mix_str = ""
        if len(active) > 1 and comp_total > 0:
            shares = " ".join(f"{k} {100 * v / comp_total:.0f}%"
                              for k, v in sorted(active.items(), key=lambda kv: -abs(kv[1])))
            loss_mix_str = f" | loss mix: {shares}"

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train RMSE: {train_metrics['RMSE']:.4f} | "
              f"Val RMSE: {val_metrics['RMSE']:.4f} | "
              f"Val Hs_MAPE: {hs_mape_str} | "
              f"Val CC: {val_metrics['CC']:.4f} | "
              f"Val {objective_metric}: {val_score:.4f} (smoothed: {smoothed_score:.4f}) | "
              f"tf_ratio: {tf_ratio:.2f}"
              + loss_mix_str
              + bulk_str)

        if trial is not None:
            trial.report(smoothed_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if smoothed_score > best_val_score:
            best_val_score = smoothed_score
            best_val_metrics = val_metrics
            # Carry the selected epoch's training-loss composition alongside
            # the validation metrics, so objective() can record it without a
            # signature change. Prefixed to keep it distinguishable from
            # evaluate()'s own keys, which are all validation-side.
            if components:
                best_val_metrics = {**val_metrics,
                                    **{f'train_loss_share_{k}': v for k, v in components.items()}}
            # Snapshot the weights at this epoch so downstream evaluation uses
            # the best checkpoint, not whatever the last epoch produced.
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping")
                break

    return best_val_score, best_val_metrics, best_model_state


def _prepare_dataloaders(density, alpha_1, alpha_2, r_1, r_2, seq_len, lead_time, batch_size,
                          target, shuffle_seed, wind=None, channel_set='full', aux_set='none'):
    """Split, normalise, and window the spectral (+ optional aux) channels into DataLoaders.

    Shared by objective() and scripts/train.py so the train/val/test split and
    normalisation are always computed identically for a given (seq_len,
    lead_time, target) — a final retrain must see exactly the same splits the
    hyperparameter search saw for its metrics to be comparable.

    channel_set selects which frequency-resolved channels (nn.channels.CHANNEL_SETS)
    are stacked into the encoder input via prepare_X. aux_set selects which
    scalar-per-timestep side channels (nn.channels.AUX_CHANNEL_SETS, e.g. wind)
    are fused into the encoder separately via prepare_aux — wind is required
    (non-None) whenever aux_set != 'none'.

    Returns (train_loader, val_loader, test_loader, freq_means, shape_means,
    num_freqs, num_channels, num_aux_channels).
    freq_means is the per-frequency training-split mean μ(f) — the
    denormalisation key E_phys = Ẽ * μ(f) used throughout training/eval, and
    (for target == 'density') the log-space floor reference (see
    utils.to_log_space). shape_means is the per-frequency training-split
    mean of the physical unit-area shape target — the analogous log-space
    floor reference for target == 'shape'; None for other targets.
    """
    n = len(density)
    train_end = int(0.7 * n)   # 70% train
    val_end   = int(0.85 * n)  # 15% val + 15% test

    train_density = density[:train_end]
    val_density   = density[train_end:val_end]
    test_density  = density[val_end:]

    train_alpha1, val_alpha1, test_alpha_1 = alpha_1[:train_end], alpha_1[train_end:val_end], alpha_1[val_end:]
    train_alpha2, val_alpha2, test_alpha_2 = alpha_2[:train_end], alpha_2[train_end:val_end], alpha_2[val_end:]
    train_r1, val_r1, test_r1             = r_1[:train_end], r_1[train_end:val_end], r_1[val_end:]
    train_r2, val_r2, test_r2             = r_2[:train_end], r_2[train_end:val_end], r_2[val_end:]

    # Compute per-frequency training mean μ(f) BEFORE normalising.
    # This tensor is the denormalisation key: E_phys = Ẽ * μ(f).
    # It is passed to the training loop and evaluator so that all spectral
    # integrations (Hs, Tm02, shape error, SI) and the density training loss
    # operate on physical m² Hz⁻¹ values, not on normalised dimensionless ones.
    freq_means = torch.tensor(
        train_density.mean().clip(lower=1e-8).values, dtype=torch.float32
    )  # shape: (num_freqs,)

    # For the Hs and shape targets, build sequence targets from PHYSICAL
    # (pre-normalisation) density so that y_batch and the persistence start
    # token are both physically meaningful (metres for hs; a true unit-area
    # shape for 'shape' — freq-mean scale normalisation would otherwise
    # distort the shape, since it scales each bin by a different constant).
    # For the density target, targets are the normalised spectra (model operates
    # in normalised space; freq_means is applied externally at loss/metric time).
    shape_means = None
    val_m0 = test_m0 = None
    if target == 'hs':
        train_y = prepare_y(train_density, seq_len, lead_time, target='hs')
        val_y   = prepare_y(val_density,   seq_len, lead_time, target='hs')
        test_y  = prepare_y(test_density,  seq_len, lead_time, target='hs')
    elif target == 'shape':
        train_y = prepare_y(train_density, seq_len, lead_time, target='shape')
        val_y   = prepare_y(val_density,   seq_len, lead_time, target='shape')
        test_y  = prepare_y(test_density,  seq_len, lead_time, target='shape')
        # Per-frequency training-mean of the physical shape target — the
        # log-space floor reference for target == 'shape' (see
        # utils.to_log_space), fit on the training split only, same
        # discipline as freq_means above.
        shape_means = torch.clamp(train_y.mean(dim=(0, 1)), min=1e-8).to(dtype=torch.float32)
        # prepare_y discards m0 for the shape target, but evaluate()'s
        # wind-sea/swell labels need the physical spectrum (decision 029):
        # m0 = (Hs/4)^2 per target step, on the same physical split density.
        val_m0  = ((prepare_y(val_density,  seq_len, lead_time, target='hs')[..., 0] / 4) ** 2).numpy()
        test_m0 = ((prepare_y(test_density, seq_len, lead_time, target='hs')[..., 0] / 4) ** 2).numpy()

    # Normalize inputs — fit on training data, apply to all splits.
    # Density uses scale-only normalization (divide by per-frequency training mean)
    # to preserve non-negativity: compute_hs calls sqrt(trapz(density)) and would
    # produce NaN if density went negative from z-scoring. Density is always
    # normalised regardless of channel_set since it's also used for the
    # density-target y and is always included in every CHANNEL_SETS entry.
    train_density, val_density, test_density = _normalize(
        train_density, val_density, test_density, mode=NORM_MODES['density'])

    # alpha_1/alpha_2 are circular (mean/principal wave direction, degrees) —
    # decompose into sin/cos pairs so the model sees a continuous embedding
    # where e.g. 1deg and 359deg are adjacent rather than z-scored raw angles
    # that would place them at opposite extremes.
    train_alpha1_sin, val_alpha1_sin, test_alpha1_sin = (np.sin(np.radians(df)) for df in (train_alpha1, val_alpha1, test_alpha_1))
    train_alpha1_cos, val_alpha1_cos, test_alpha1_cos = (np.cos(np.radians(df)) for df in (train_alpha1, val_alpha1, test_alpha_1))
    train_alpha2_sin, val_alpha2_sin, test_alpha2_sin = (np.sin(np.radians(df)) for df in (train_alpha2, val_alpha2, test_alpha_2))
    train_alpha2_cos, val_alpha2_cos, test_alpha2_cos = (np.cos(np.radians(df)) for df in (train_alpha2, val_alpha2, test_alpha_2))

    # r1/r2 have no downstream physical constraint so z-score is safe. Only
    # normalise the ones this channel_set actually needs.
    channel_names = CHANNEL_SETS[channel_set]
    raw_channels = {
        'density':     (train_density, val_density, test_density),
        'alpha_1_sin': (train_alpha1_sin, val_alpha1_sin, test_alpha1_sin),
        'alpha_1_cos': (train_alpha1_cos, val_alpha1_cos, test_alpha1_cos),
        'alpha_2_sin': (train_alpha2_sin, val_alpha2_sin, test_alpha2_sin),
        'alpha_2_cos': (train_alpha2_cos, val_alpha2_cos, test_alpha2_cos),
        'r_1':         (train_r1, val_r1, test_r1),
        'r_2':         (train_r2, val_r2, test_r2),
    }
    normalized = {'density': (train_density, val_density, test_density)}
    for name in channel_names:
        if name == 'density':
            continue
        normalized[name] = _normalize(*raw_channels[name], mode=NORM_MODES[name])

    train_X = prepare_X([normalized[name][0] for name in channel_names], seq_len, lead_time)
    val_X   = prepare_X([normalized[name][1] for name in channel_names], seq_len, lead_time)
    test_X  = prepare_X([normalized[name][2] for name in channel_names], seq_len, lead_time)
    num_channels = len(channel_names)

    if target == 'density':
        train_y = prepare_y(train_density, seq_len, lead_time, target='density')
        val_y   = prepare_y(val_density,   seq_len, lead_time, target='density')
        test_y  = prepare_y(test_density,  seq_len, lead_time, target='density')

    # Auxiliary side-input — scalar-per-timestep, not frequency-resolved, so
    # it bypasses prepare_X/FreqDimEmbedding and is fused into the encoder
    # separately (see WaveHeightBaselineNN). Two independent sources:
    # 'wind' varies per-timestep (windowed by prepare_aux from a
    # full-length series); 'dmd' is computed ONCE per sample from that
    # sample's own already-windowed density history (nn/prepare_dmd.py),
    # then broadcast across seq_len to match prepare_aux's output shape —
    # prepare_aux itself can't compute this (it only windows an
    # already-fully-computed per-timestep series, the opposite order DMD
    # needs), hence the separate branch below.
    aux_names = AUX_CHANNEL_SETS[aux_set]
    num_aux_channels = len(aux_names)
    if aux_set == 'wind':
        if wind is None:
            raise ValueError(f"aux_set={aux_set!r} requires a wind dataframe")
        train_wind = wind[:train_end]
        val_wind   = wind[train_end:val_end]
        test_wind  = wind[val_end:]
        normalized_aux = {
            name: _normalize(train_wind[[name]], val_wind[[name]], test_wind[[name]],
                              mode=AUX_NORM_MODES[name])
            for name in aux_names
        }
        train_aux = prepare_aux([normalized_aux[name][0][name] for name in aux_names], len(train_density), seq_len, lead_time)
        val_aux   = prepare_aux([normalized_aux[name][1][name] for name in aux_names], len(val_density),   seq_len, lead_time)
        test_aux  = prepare_aux([normalized_aux[name][2][name] for name in aux_names], len(test_density),  seq_len, lead_time)
    elif aux_set == 'dmd':
        train_dmd_raw = compute_dmd_features(train_X[..., 0].numpy())
        val_dmd_raw   = compute_dmd_features(val_X[..., 0].numpy())
        test_dmd_raw  = compute_dmd_features(test_X[..., 0].numpy())
        train_dmd_df, val_dmd_df, test_dmd_df = (
            pd.DataFrame(arr, columns=aux_names)
            for arr in (train_dmd_raw, val_dmd_raw, test_dmd_raw)
        )
        normalized_aux = {
            name: _normalize(train_dmd_df[[name]], val_dmd_df[[name]], test_dmd_df[[name]],
                              mode=AUX_NORM_MODES[name])
            for name in aux_names
        }
        train_arr = np.stack([normalized_aux[name][0][name].values for name in aux_names], axis=1)
        val_arr   = np.stack([normalized_aux[name][1][name].values for name in aux_names], axis=1)
        test_arr  = np.stack([normalized_aux[name][2][name].values for name in aux_names], axis=1)
        train_aux = torch.from_numpy(np.repeat(train_arr[:, None, :], seq_len, axis=1).astype(np.float32))
        val_aux   = torch.from_numpy(np.repeat(val_arr[:, None, :],   seq_len, axis=1).astype(np.float32))
        test_aux  = torch.from_numpy(np.repeat(test_arr[:, None, :],  seq_len, axis=1).astype(np.float32))
    else:
        train_aux = prepare_aux([], len(train_density), seq_len, lead_time)
        val_aux   = prepare_aux([], len(val_density),   seq_len, lead_time)
        test_aux  = prepare_aux([], len(test_density),  seq_len, lead_time)

    # DataLoaders — generator seeded explicitly so shuffle order is reproducible
    # for a given shuffle_seed (trial.number during HPO; a chosen seed during
    # final retrain).
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(WaveSpectralDataset(train_X, train_aux, train_y), batch_size=batch_size, shuffle=True,
                              worker_init_fn=_seed_worker, generator=g)
    val_loader   = DataLoader(WaveSpectralDataset(val_X, val_aux, val_y, m0_true=val_m0), batch_size=batch_size, shuffle=False,
                              worker_init_fn=_seed_worker, generator=g)
    test_loader  = DataLoader(WaveSpectralDataset(test_X, test_aux, test_y, m0_true=test_m0), batch_size=batch_size, shuffle=False,
                              worker_init_fn=_seed_worker, generator=g)

    return (train_loader, val_loader, test_loader, freq_means, shape_means,
            train_X.shape[2], num_channels, num_aux_channels)


def objective(trial, *, density, alpha_1, alpha_2, r_1, r_2, freqs, lead_time, target,
              objective_metric='weighted_mean_SS', results_folder=None,
              wind=None, channel_set='full', aux_set='none',
              base_loss_weight=1.0, compute_peak_metrics=False,
              fixed_head_dim=None, fixed_nhead=None):
    """
    fixed_head_dim, fixed_nhead : int | None, default None (search both as
        before — fully backward compatible). See their comment at the
        head_dim/nhead sampling site below for the cross-version tally
        motivating v13's choice to pin both (32, 8) for target == 'shape'.
    base_loss_weight : float, default 1.0 — forwarded to _train_model/
        train_one_epoch unchanged (see that docstring). Left at 1.0 (the
        per-bin loss trains normally) for every target/study EXCEPT the
        v13 KL+Wasserstein+Peak composite-loss search (scripts/optimize.py),
        which passes 0.0 to literally substitute it, per
        scripts/ablate_loss.py's validated 'combined' recipe — this is
        NOT validated for target in ('hs', 'density'), so callers for
        those targets must not pass 0.0 here.
    compute_peak_metrics : bool, default False — forwarded to
        _train_model/evaluate(). Must be True whenever objective_metric ==
        'peak_fidelity' (see _compute_val_score's docstring) — that
        metric needs the wind-sea/swell panel evaluate() only computes
        when this flag is set.
    """
    # set_seed() is called once at script level — do NOT call it here.
    # Resetting the RNG inside objective() makes every trial start from the same
    # random state, collapsing the variance Optuna needs to learn from.

    # Sample hyperparameters
    seq_len = trial.suggest_categorical('seq_len', [12, 24, 48, 96])
    # The scheduled-sampling training loop (train_one_epoch) unrolls one
    # forward pass per decoder step and only backpropagates once at the end,
    # so its memory footprint grows ~quadratically with lead_time. Cap
    # batch_size for longer horizons to avoid CUDA OOM
    batch_size_choices = [32, 64] if lead_time > 24 else [32, 64, 128]
    batch_size = trial.suggest_categorical('batch_size', batch_size_choices)
    # Narrowed to bracket shape_v9's best trials (lr 3.7e-3 - 9.2e-3 across
    # lead times) with headroom, now that we have a region to focus on.
    #
    # v12 caveat: this bracket was tuned under the OLD Softplus +
    # physical-space-RMSE regime for 'density'/'shape' targets. Those targets
    # now train on frequency-weighted plain MSE in log-space (see
    # scripts/optimize.py's STUDY_VERSION v12 comment) — a comparably large
    # regime change to the Adam->AdamW switch that this file's weight_decay
    # comment (below) deliberately did NOT re-narrow for. This range is left
    # as-is for now (not silently assumed valid) — widen it if v12 trials
    # cluster at either edge.
    lr = trial.suggest_float('lr', 1e-3, 1.5e-2, log=True)
    # Split by which representation width the dropout acts on, rather than by
    # module identity: freq_embed_dropout regularizes the freq_embed_dim=8
    # per-bin representation inside every FreqDimEmbedding instance (encoder's
    # and decoder's, when present), while embed_dropout regularizes the wider
    # embed_dim representation shared by PositionalEncoding, the top-level
    # (time-axis) TemporalConvFrontend, and nn.Transformer's own internal
    # self-attention/FFN dropout. Dropping units out of the narrow 8-wide
    # representation is a much bigger relative perturbation than dropping
    # units out of the >=16-wide embed_dim representation, so the two
    # plausibly want different optima.
    freq_embed_dropout = trial.suggest_float('freq_embed_dropout', 0.1, 0.3)
    # Lower bound 0.0 (vs freq_embed_dropout's 0.1) because this now also
    # covers nn.Transformer's own dropout, which was previously never wired
    # up at all and silently stuck at the library default of 0.1 — see
    # nn/transformer.py's nn.Transformer(...) construction.
    embed_dropout = trial.suggest_float('embed_dropout', 0.0, 0.3)
    # embed_dim derived as head_dim × nhead so it is always divisible by nhead.
    # nhead starts at 4 so the minimum embed_dim is 8×4=32.
    #
    # fixed_head_dim/fixed_nhead (v13, new, both default None = search as
    # before): a cross-version tally of every completed shape-target study
    # sharing this search-space shape (shape_v10/v11's best_trial.txt,
    # shape_v12's current_best.txt — 8 (study, lead_time) data points total)
    # found head_dim=32 winning 6/8 and nhead=8 winning 6/8, vs. no other
    # hyperparameter in this function showing anywhere near that level of
    # cross-lead-time/cross-version agreement (seq_len/batch_size/dropouts/
    # weight_decay each span nearly their entire range with no consensus
    # value — see results/lossablation_comparison_v2.md-adjacent analysis
    # for the full tally). NOT unanimous (shape_v12 itself picked 8/16/32
    # across its own 3 lead times) and NOT extended to num_encoder_layers
    # despite a similar-looking 5/8 for value 4 — that one's dissenting
    # picks include a lead_time (shape_v10's 6h) picking a very different
    # value (1), suggesting genuine lead-time-dependent capacity need
    # rather than noise, unlike head_dim/nhead's dissents.
    if fixed_head_dim is not None:
        head_dim = fixed_head_dim
    else:
        head_dim = trial.suggest_categorical('head_dim', [8, 16, 32])
    if fixed_nhead is not None:
        nhead = fixed_nhead
    else:
        nhead = trial.suggest_categorical('nhead', [4, 8])
    embed_dim = head_dim * nhead
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 4)
    # NOT narrowed around shape_v9's best weight_decay values (5.6e-5 - 4.4e-4),
    # despite lr being narrowed above: those values were tuned under optim.Adam,
    # which applies weight_decay as L2 regularization coupled into the gradient
    # (then scaled by Adam's per-parameter adaptive moment estimates), whereas
    # AdamW (see _train_model) decouples it into a direct
    # param -= lr * weight_decay * param step. The two are not known to share
    # an optimal region, so this keeps the original wide range to let Optuna
    # re-discover it under the new optimizer.
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    # target == 'shape' only (no-op otherwise — see train_one_epoch).
    #
    # v14: the three loss weights are no longer sampled as raw multipliers.
    # train_one_epoch sums the terms RAW and they live in four unrelated units
    # spanning ~4.5 orders of magnitude (see LOSS_TERM_REFERENCE), so a raw
    # weight carries no information about how much its term contributes. v13
    # sampled them independently over 1-200 / 0.1-50 / 0.01-20, which — once
    # converted to contributions — put the MEDIAN draw at roughly 350:1
    # peak-dominated and left KL and W2 nearly inert, with a worst case near
    # 6e5:1. That is the regime decision 032 showed is degenerate for the peak
    # term, which is position-blind and constrains only ~2.5 scalars of a
    # 47-dim output. The search was not merely exposed to it; it was centred
    # in it (decision 033).
    #
    # Instead: one overall scale, and two ratios ANCHORED ON KL. With
    # base_loss_weight=0, KL is the only full-support, position-determining
    # term left in the composite, so expressing the other two relative to it
    # is what structurally bounds how far a draw can drift from a
    # position-determined loss — worst case 20:1 rather than 6e5:1.
    #
    # Calibration: the lossablation_v3 'peak' arm (the best-scoring arm at the
    # time of writing) sits at peak_rel ~= 11.7 and is well behaved, while
    # 'peak_only' (peak_rel -> infinity, KL absent) collapses. So the evidence
    # puts the safe ceiling above 11.7, not below it; 20 leaves headroom
    # without reopening the degenerate regime. The 0.02 floor lets TPE
    # discover that a term isn't needed without ever reaching exactly zero,
    # which for the peak term IS the degenerate direction.
    #
    # kl_contrib doubles as the loss's overall scale — with no base term to
    # balance against it behaves like an effective LR multiplier, as
    # scripts/ablate_loss.py's 'kl' phase comment notes. Gradient clipping
    # (max_norm=1.0, train_one_epoch) bounds the top end.
    kl_contrib = trial.suggest_float('kl_contrib', 0.05, 5.0, log=True)
    w2_rel = trial.suggest_float('w2_rel', 0.02, 20.0, log=True)
    peak_rel = trial.suggest_float('peak_rel', 0.02, 20.0, log=True)
    # Derived, not sampled — resolve_loss_weights() is the single definition of
    # this mapping, so scripts/train.py reconstructs identical weights from a
    # saved best_trial.txt without duplicating the arithmetic.
    _loss_weights = resolve_loss_weights(
        {'kl_contrib': kl_contrib, 'w2_rel': w2_rel, 'peak_rel': peak_rel})
    kl_loss_weight = _loss_weights['kl_loss_weight']
    wasserstein_loss_weight = _loss_weights['wasserstein_loss_weight']
    peak_loss_weight = _loss_weights['peak_loss_weight']

    # Safety net: embed_dim must be divisible by nhead (guaranteed by construction
    # above, but kept to catch any future reparameterization changes).
    if embed_dim % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    # --- Data preparation ---
    # DataLoader shuffle seeded per trial (using trial.number) so shuffle order
    # differs between trials while remaining reproducible within each trial.
    train_loader, val_loader, test_loader, freq_means, shape_means, num_freqs, num_channels, num_aux_channels = (
        _prepare_dataloaders(
            density, alpha_1, alpha_2, r_1, r_2, seq_len, lead_time, batch_size, target,
            shuffle_seed=trial.number, wind=wind, channel_set=channel_set, aux_set=aux_set)
    )

    # --- Model ---
    device = get_device()
    print(f'Running on device: {device}')

    model = WaveHeightBaselineNN(
        num_freqs=num_freqs,
        freqs=freqs,
        target=target,
        num_channels=num_channels,
        num_aux_channels=num_aux_channels,
        freq_embed_dropout=freq_embed_dropout,
        embed_dropout=embed_dropout,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        embed_dim=embed_dim,
    )
    model = model.to(device)

    try:
        best_val_score, best_val_metrics, best_model_state = _train_model(
            model, train_loader, val_loader, device, freqs, freq_means, shape_means,
            target, lead_time, lr, weight_decay, objective_metric,
            num_epochs=100, patience=20, trial=trial,
            base_loss_weight=base_loss_weight,
            kl_loss_weight=kl_loss_weight,
            wasserstein_loss_weight=wasserstein_loss_weight,
            peak_loss_weight=peak_loss_weight,
            peak_max_count=4,  # fixed, not searched — see scripts/ablate_loss.py's MAX_PEAKS
            compute_peak_metrics=compute_peak_metrics)
    except torch.OutOfMemoryError:
        # Drop references to this trial's model/optimizer/activations before
        # emptying the cache — otherwise the exception's traceback keeps the
        # frame (and its tensors) alive and the freed memory never reaches
        # the allocator, starving the next trial too.
        del model
        empty_cache(device)
        raise

    # Checkpoint the model whenever this trial beats every trial completed so
    # far, mirroring how save_progress overwrites current_best.txt. Trials run
    # sequentially (no n_jobs>1), so trial.study.best_value at this point still
    # reflects only trials 0..(this one - 1) — exactly "did I just become the
    # new best". Same checkpoint format as scripts/train.py's final retrain so
    # either can be loaded the same way later.
    if results_folder is not None and best_model_state is not None:
        try:
            current_best = trial.study.best_value
        except ValueError:
            current_best = float('-inf')
        if best_val_score > current_best:
            # trial.params only holds what was actually sampled via
            # trial.suggest_*, so when fixed_head_dim/fixed_nhead pin these
            # instead of searching them, 'head_dim'/'nhead' are silently
            # absent from it -- breaking nn.checkpoints.build_model (and
            # everything built on it: scripts/compare_versions.py,
            # scripts/infer.py), which reads embed_dim as
            # params['head_dim'] * params['nhead'] unconditionally. Folding
            # the actual (fixed-or-searched) values in here keeps 'params'
            # a complete reconstruction recipe regardless of which path set
            # them, matching every other checkpoint's shape.
            saved_params = {**trial.params, 'head_dim': head_dim, 'nhead': nhead}
            torch.save({
                'model_state_dict': best_model_state,
                'params': saved_params,
                'target': target,
                'lead_time_steps': lead_time,
                'freq_means': freq_means,
                'shape_means': shape_means,
                'freqs': freqs,
                'trial_number': trial.number,
                'val_score': best_val_score,
            }, Path(results_folder) / 'best_model.pt')

    # Store all validation metrics from the best epoch as trial user attributes
    if best_val_metrics is not None:
        scalar_keys = ['RMSE', 'Hs_MAPE', 'CC', 'Bias', 'R2', 'overall_SS']
        list_keys = ['per_step_RMSE', 'per_step_RMSE_pers', 'per_step_SS', 'per_step_Bias', 'per_step_R2']
        for key in scalar_keys:
            trial.set_user_attr(f'val_{key}', best_val_metrics[key])
        for key in list_keys:
            trial.set_user_attr(f'val_{key}', best_val_metrics[key])
        if target == 'density':
            for key in ['Hs_RMSE', 'Hs_Bias', 'Tm02_RMSE', 'Tm02_Bias',
                        'Shape_masked_samples', 'SI_per_bin', 'SI_mean']:
                if key in best_val_metrics:
                    trial.set_user_attr(f'val_{key}', best_val_metrics[key])
        if target in ('density', 'shape'):
            # Shape_RMSE/Shape_SS are computed for both target types (see
            # nn/evaluate.py); Shape_Mass_Error only for 'shape'.
            for key in ['Shape_RMSE', 'Shape_SS', 'Shape_Mass_Error']:
                if key in best_val_metrics:
                    trial.set_user_attr(f'val_{key}', best_val_metrics[key])
        # The raw inputs of 'peak_fidelity' (present only when the run passed
        # compute_peak_metrics=True). Stored so a later revision of the
        # metric can re-rank a finished study offline: shape_v13 selected on
        # the pre-030 score kept none of these, so its 80 trials per lead had
        # to be thrown away rather than re-scored when the detector was fixed
        # -- trial.value alone is the old number and nothing else survives.
        # Peak_Count_True_Mean/_Pred_Mean are in the list because that pair,
        # not the score, is what makes an over- or under-segmenting model
        # obvious at a glance.
        for key in ['Peak_Separation_Recall_windsea', 'Peak_Separation_Recall_swell',
                    'Peak_Separation_Precision',
                    'Peak_Height_RelError_windsea', 'Peak_Height_RelError_swell',
                    'Peak_Count_True_Mean', 'Peak_Count_Pred_Mean',
                    'Peak_windsea_n', 'Peak_swell_n',
                    'Tm02_RMSE_windsea', 'Tm02_RMSE_swell',
                    'Tm02_Bias_windsea', 'Tm02_Bias_swell']:
            if key in best_val_metrics:
                trial.set_user_attr(f'val_{key}', best_val_metrics[key])
        # Training-loss composition at the selected epoch (see _train_model).
        # Stored unprefixed by 'val_' because these are training-side, and
        # stored at all because a weight's numeric value says nothing about
        # how much its term contributed -- the ratios are the only way to see
        # a trial that ran dominated by one term (decisions 032, 033).
        for key in ('base', 'w2', 'kl', 'peak'):
            share_key = f'train_loss_share_{key}'
            if share_key in best_val_metrics:
                trial.set_user_attr(share_key, best_val_metrics[share_key])

    # Restore best-epoch weights before test evaluation so the reported test
    # metrics correspond to the same model that produced best_val_score.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, freqs,
                             lead_time=lead_time, freq_means=freq_means,
                             shape_means=shape_means)
    print(f"Final test metrics: {test_metrics}")
    return best_val_score
