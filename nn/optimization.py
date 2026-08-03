import math
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
# deliberately NOT tied to the trial's own (tunable) wasserstein_loss_weight
# hyperparameter. Using the trial's own weight here would make cross-trial
# comparison unfair: trials sampling a large wasserstein_loss_weight would
# get a structurally different scoring scale than trials sampling a small
# one, purely from that hyperparameter choice, corrupting the very
# comparison Optuna's search depends on. This is a separate, constant
# analyst choice instead.
#
# Value chosen by rough order-of-magnitude matching against observed test-set
# ranges from the manual validation (not a precisely fit constant — revisit
# once real study data exists): final_step_SS-family metrics sit around
# 0.1-0.2 in this problem (e.g. best_val_weighted_mean_SS was 0.17-0.18
# across the manually-tested configs), while Shape_Wasserstein sits around
# 0.011-0.014 for well-trained models. BETA=10 puts a typical
# Shape_Wasserstein contribution (~0.1-0.14) on a comparable scale to a
# typical SS value, so neither term structurally dominates the other by
# construction alone.
_FINAL_STEP_SS_WASSERSTEIN_BETA = 10.0


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

    All metrics are transformed so that higher = better, matching Optuna's
    'maximize' direction:
    - Skill Scores are already higher-is-better.
    - Error metrics (RMSE, Hs_RMSE, etc.) are negated.

    Valid values for objective_metric:
        'final_step_SS'    : Skill Score at the last forecast step only (i.e.
                             the actual chosen lead time — steps before it are
                             autoregressive scaffolding, not a deliverable in
                             their own right)
        'weighted_mean_SS' : exponentially-weighted mean per-step Skill Score
                             (robust to variable seq_len, but biases toward
                             the earlier/easier steps rather than the step
                             that's actually forecast)
        'overall_SS'       : Skill Score on flattened all-step RMSE
        'Hs_SS'            : Hs Skill Score — robust to seq_len variation and
                             directly targets Hs. For target=='hs' equals
                             overall_SS; for target=='density' computed from
                             denormalised spectra (see evaluate.py).
        'RMSE'             : negative overall RMSE
        'Hs_RMSE'          : negative Hs RMSE (density target only)
        'Tm02_RMSE'        : negative Tm02 RMSE (density target only)
        'Shape_RMSE'       : negative spectral shape RMSE (density target only)
        'SI_mean'          : negative mean Scatter Index (density target only)
        'final_step_SS_wasserstein' : final_step_SS minus a fixed penalty on
                             Shape_Wasserstein (shape target only, for now —
                             Shape_Wasserstein is only computed in that
                             target's evaluate() block). See
                             _FINAL_STEP_SS_WASSERSTEIN_BETA's comment for why
                             this exists and why its weight is NOT the same
                             as the training-time wasserstein_loss_weight.
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
    else:
        raise ValueError(
            f"Unknown objective_metric {objective_metric!r}. Valid: "
            "'final_step_SS', 'weighted_mean_SS', 'overall_SS', 'Hs_SS', "
            "'RMSE', 'Hs_RMSE', 'Tm02_RMSE', 'Shape_RMSE', 'SI_mean', "
            "'final_step_SS_wasserstein'"
        )


def _train_model(model, train_loader, val_loader, device, freqs, freq_means,
                  shape_means, target, lead_time, lr, weight_decay, objective_metric,
                  num_epochs=80, patience=10, trial=None, wasserstein_loss_weight=0.0):
    """Run the scheduled-sampling training loop with early stopping.

    Shared by objective() (Optuna trial) and scripts/train.py (fixed-config
    final retrain) so the two never drift apart. When `trial` is given,
    reports the per-epoch score to Optuna and prunes on its signal; this is
    the only behavioural difference between the two callers.

    wasserstein_loss_weight : float, target == 'shape' only, default 0.0 (no
        behavior change) — forwarded to train_one_epoch's auxiliary
        SpectralWassersteinLoss term (see nn/training_loop.py docstring).
        Not yet part of objective()'s Optuna search space — currently only
        scripts/train.py sets this to a nonzero value, for a manual
        before/after comparison ahead of adding it as a tunable hyperparameter.

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

    # Linear LR warmup: this transformer has no warmup and trains AdamW at the
    # sampled lr from epoch 0, which is known to be unstable for transformers
    # at the higher end of a search range. Optuna analysis of shape_v11/v12
    # found lr dominating hyperparameter importance (0.23-0.69) while only
    # weakly correlating with score (0.23-0.47) — the best and worst trials at
    # every lead time drew lr from almost the same range, the signature of
    # noisy/unstable early training rather than a clean optimum TPE can
    # exploit. Ramping up to the sampled lr over the first few epochs should
    # cut that instability and let the true hyperparameter signal (lr's own
    # and everyone else's) come through more cleanly. ReduceLROnPlateau only
    # starts stepping once warmup ends, so the ramp itself is never mistaken
    # for a plateau.
    WARMUP_EPOCHS = 5

    # tf_ratio decays from 1.0 to 0.0 over 2×patience epochs.  With early
    # stopping at patience=20, a run going to epoch ~40 will have tf_ratio
    # ≈ 0.5 — half its training steps use the model's own predictions, which
    # meaningfully closes the teacher-forcing / autoregressive distribution gap.
    tf_decay_epochs = 2 * patience

    # Epoch-to-epoch val_score is noisy (autoregressive eval on a small val
    # split) — this trailing mean smooths it before it drives ANY decision:
    # the LR scheduler, early-stopping/checkpoint selection, AND (when
    # running as an Optuna trial) the pruner report. Originally only the
    # pruner report was smoothed (best_val_score/early-stopping used the raw
    # per-epoch value on the reasoning that picking a checkpoint was lower-
    # stakes than pruning a whole trial) — that assumption broke for the
    # 'final_step_SS_wasserstein' objective_metric: its Shape_Wasserstein
    # term has enough of its own per-epoch variance (amplified by
    # _FINAL_STEP_SS_WASSERSTEIN_BETA) that an unsmoothed run locked its best
    # checkpoint onto an early noise spike (epoch 14) and never recognized
    # genuinely continued improvement in later epochs (Shape_SS climbing
    # steadily through epoch 30+) as "better" — early stopping then fired on
    # a stale, undertrained checkpoint. Smoothing everything from the same
    # trailing window removes that failure mode instead of just the pruner's
    # narrower version of it.
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
                                        wasserstein_loss_weight=wasserstein_loss_weight)
        val_metrics   = evaluate(model, val_loader, device, freqs,
                                  lead_time=lead_time, freq_means=freq_means,
                                  shape_means=shape_means)

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
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train RMSE: {train_metrics['RMSE']:.4f} | "
              f"Val RMSE: {val_metrics['RMSE']:.4f} | "
              f"Val Hs_MAPE: {hs_mape_str} | "
              f"Val CC: {val_metrics['CC']:.4f} | "
              f"Val {objective_metric}: {val_score:.4f} (smoothed: {smoothed_score:.4f}) | "
              f"tf_ratio: {tf_ratio:.2f}"
              + bulk_str)

        if trial is not None:
            trial.report(smoothed_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if smoothed_score > best_val_score:
            best_val_score = smoothed_score
            best_val_metrics = val_metrics
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
    val_loader   = DataLoader(WaveSpectralDataset(val_X, val_aux, val_y), batch_size=batch_size, shuffle=False,
                              worker_init_fn=_seed_worker, generator=g)
    test_loader  = DataLoader(WaveSpectralDataset(test_X, test_aux, test_y), batch_size=batch_size, shuffle=False,
                              worker_init_fn=_seed_worker, generator=g)

    return (train_loader, val_loader, test_loader, freq_means, shape_means,
            train_X.shape[2], num_channels, num_aux_channels)


def objective(trial, *, density, alpha_1, alpha_2, r_1, r_2, freqs, lead_time, target,
              objective_metric='weighted_mean_SS', results_folder=None,
              wind=None, channel_set='full', aux_set='none'):
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
    head_dim = trial.suggest_categorical('head_dim', [8, 16, 32])
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
    # target == 'shape' only (no-op otherwise — see train_one_epoch). Range
    # from a manual sweep (not committed) reusing shape_v11's exact other
    # hyperparameters: weights 50 and 150 both improved every metric
    # (Shape_RMSE/SS, peak-separation recall, unimodal AND multimodal SS)
    # monotonically over the 0-weight baseline, with no sign of diminishing
    # returns yet at 150 — the upper bound here (400) is deliberately well
    # above the highest manually-tested value rather than assuming 150 was
    # near-optimal; the lower bound (10) sits below 50 (the smallest value
    # that showed a real effect) so Optuna can still discover "less matters"
    # across the wider hyperparameter space this searches vs. the manual
    # sweep's fixed other-hyperparameters test.
    wasserstein_loss_weight = trial.suggest_float('wasserstein_loss_weight', 10.0, 400.0, log=True)

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
            wasserstein_loss_weight=wasserstein_loss_weight)
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
            torch.save({
                'model_state_dict': best_model_state,
                'params': trial.params,
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

    # Restore best-epoch weights before test evaluation so the reported test
    # metrics correspond to the same model that produced best_val_score.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, freqs,
                             lead_time=lead_time, freq_means=freq_means,
                             shape_means=shape_means)
    print(f"Final test metrics: {test_metrics}")
    return best_val_score
