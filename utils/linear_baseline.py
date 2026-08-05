"""Simple linear AR baseline — a second sanity check alongside persistence.

Persistence (see nn/evaluate.py) answers "does the model beat doing nothing."
This module answers a different question: does the transformer beat a plain
*linear* extrapolator of each frequency bin's own history? A model that only
narrowly beats this baseline is not obviously exploiting nonlinear structure.

Design (see CLAUDE.md discussion / plan for the full rationale):
- Recursive AR rollout: fit a one-step-ahead linear model, then feed its own
  prediction back as the newest lag for the next step, `lead_time` times —
  mirroring model.infer()'s autoregressive loop and the persistence
  baseline's "last observed value" discipline.
- Independent per-frequency-bin AR: each frequency bin (or, for 'hs', the
  scalar Hs series) is its own univariate AR(p) model using only its own
  past values — no cross-frequency/cross-channel coupling. This directly
  tests whether FreqDimEmbedding's cross-bin attention pooling is earning
  its keep.

No dependency on nn/ (utils must not import from nn — see utils/loss.py's
_FULL_CHANNELS comment for why: nn/__init__.py eagerly imports submodules
that import from utils, so the reverse import would be circular). The 70/15/15
split fractions are therefore duplicated from nn/optimization.py::
_prepare_dataloaders rather than imported.

Entry points, in typical call order:
- fit_linear_ar_from_density : fit coefficients on the train split.
- forecast_coeffs             : roll already-fitted coefficients out over a
                                 chosen split, returning raw physical arrays
                                 (pred, true, pers) — useful when a caller
                                 (e.g. scripts/compare_versions.py) wants to
                                 feed those arrays into another metrics
                                 function, such as
                                 nn/spectrum_eval.py::compute_density_metrics,
                                 for exact parity with how a transformer
                                 checkpoint's forecasts are scored there.
- evaluate_coeffs             : forecast_coeffs + this module's own metrics
                                 (for already-fitted coefficients, e.g.
                                 loaded from a checkpoint).
- evaluate_linear_ar          : fit_linear_ar_from_density + evaluate_coeffs
                                 in one call — the common case.
"""
import numpy as np
import torch

from .compute_hs import compute_hs_from_density, compute_bulk_params, compute_shape, trapz_weights
from .loss import RMSELoss

M0_MASK_THRESHOLD = 1e-4  # m² — same as nn/evaluate.py's density-target mask


def _to_numpy(freqs):
    return freqs.cpu().numpy() if torch.is_tensor(freqs) else np.asarray(freqs)


def _split_train_val_test(density):
    """Same 70/15/15 chronological split fractions as
    nn/optimization.py::_prepare_dataloaders (train_end=int(0.7*n),
    val_end=int(0.85*n)), duplicated here since utils has no dependency on nn.
    """
    n = len(density)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    return density.iloc[:train_end], density.iloc[train_end:val_end], density.iloc[val_end:]


def _derive_target_series(density_split, freqs_np, target):
    """Physical target series for a density split, same vocabulary/derivation
    as nn/prepare_y.py: 'hs' -> compute_hs (time, 1), 'shape' -> compute_shape
    (time, num_freqs), 'density' -> the raw physical values unchanged."""
    vals = density_split.values.astype(np.float64)
    if target == 'hs':
        return compute_hs_from_density(vals, freqs_np)[:, np.newaxis]  # (time, 1)
    elif target == 'shape':
        return compute_shape(vals, freqs_np)  # (time, num_freqs)
    else:  # 'density'
        return vals  # (time, num_freqs)


def fit_linear_ar(series, order, ridge=1e-6):
    """Fit an independent AR(order) model per column of `series`.

    Parameters
    ----------
    series : np.ndarray, shape (time,) or (time, num_cols) — physical units.
    order  : int, number of lags (p). Use the compared model's `seq_len` so
             both baselines see the same lookback window.
    ridge  : float, tiny L2 penalty added purely for numerical stability
             (adjacent lags / frequency bins are highly correlated, so plain
             OLS can be near-singular for large `order`) — not meaningful
             shrinkage, so this stays "a very simple linear model."

    Returns
    -------
    coeffs : np.ndarray, shape (num_cols, order + 1). coeffs[:, 0] is the
             intercept; coeffs[:, 1:] are lag weights ordered
             [x_t, x_{t-1}, ..., x_{t-order+1}] (most recent lag first).
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[:, np.newaxis]
    num_timesteps, num_cols = series.shape
    num_rows = num_timesteps - order
    if num_rows < 1:
        raise ValueError(f"series has {num_timesteps} timesteps, too short for order={order}")

    # X[i] = [1, x_{i+order-1}, x_{i+order-2}, ..., x_i] predicts x_{i+order}
    # — most-recent lag first, matching _rollout's window convention below.
    lags = np.stack([series[order - 1 - j: num_rows + order - 1 - j] for j in range(order)], axis=1)
    coeffs = np.zeros((num_cols, order + 1))
    ridge_rows = np.sqrt(ridge) * np.eye(order + 1)
    zeros_rhs = np.zeros(order + 1)
    for c in range(num_cols):
        X_c = np.concatenate([np.ones((num_rows, 1)), lags[:, :, c]], axis=1)  # (num_rows, order+1)
        y_c = series[order:, c]  # (num_rows,)
        X_aug = np.concatenate([X_c, ridge_rows], axis=0)
        y_aug = np.concatenate([y_c, zeros_rhs], axis=0)
        sol, *_ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
        coeffs[c] = sol
    return coeffs


def fit_linear_ar_from_density(density, freqs, seq_len, target, ridge=1e-6):
    """Fit AR(seq_len) coefficients on the train split only, deriving the
    target series the same way nn/prepare_y.py does.

    Returns coeffs (see fit_linear_ar) — pass to forecast_coeffs/
    evaluate_coeffs to score them (e.g. on val/test, or after loading them
    back from a checkpoint) without refitting.
    """
    freqs_np = _to_numpy(freqs)
    train_density, _, _ = _split_train_val_test(density)
    train_series = _derive_target_series(train_density, freqs_np, target)
    return fit_linear_ar(train_series, order=seq_len, ridge=ridge)


def _rollout(coeffs, last_windows, lead_time, freqs=None):
    """Recursively forecast `lead_time` steps ahead from `last_windows`.

    Parameters
    ----------
    coeffs       : np.ndarray, shape (num_cols, order + 1), from fit_linear_ar.
    last_windows : np.ndarray, shape (num_samples, order, num_cols) — the most
                   recent `order` physical observations before each forecast
                   start time, most-recent-last (i.e. last_windows[:, -1, :]
                   is the last observed value — same convention as
                   prepare_X/get_start_token).
    lead_time    : int, number of steps to forecast.
    freqs        : np.ndarray | None, shape (num_freqs,). When given (i.e.
                   target == 'shape'), each step's per-bin prediction is
                   renormalized to unit area via compute_shape before being
                   recorded/fed back — mirrors model.infer()'s per-step
                   unit-area renormalization for the 'shape' target.

    Returns
    -------
    forecasts : np.ndarray, shape (num_samples, lead_time, num_cols).
    """
    num_samples, order, num_cols = last_windows.shape
    intercept = coeffs[:, 0]          # (num_cols,)
    weights = coeffs[:, 1:]           # (num_cols, order), most-recent-first

    # window[:, k, :] holds lag (order-k), i.e. window[:, -1, :] is the most
    # recent observation — reversed once here so it aligns with `weights`'
    # most-recent-first ordering without re-reversing every step.
    window = last_windows[:, ::-1, :].copy()  # (num_samples, order, num_cols)

    forecasts = np.empty((num_samples, lead_time, num_cols))
    for step in range(lead_time):
        # (num_samples, order, num_cols) * (num_cols, order) -> (num_samples, num_cols)
        pred = intercept[np.newaxis, :] + np.einsum('sok,ko->sk', window, weights)
        if freqs is not None:
            pred = compute_shape(pred, freqs)
        forecasts[:, step, :] = pred
        window = np.concatenate([pred[:, np.newaxis, :], window[:, :-1, :]], axis=1)

    return forecasts


def _make_windows(series, order, lead_time):
    """Build (lag windows, targets) exactly matching prepare_X/prepare_y's
    indexing: num_samples = num_timesteps - order - lead_time + 1.

    Returns
    -------
    windows : np.ndarray, shape (num_samples, order, num_cols) — most recent
              observation last (window[:, -1, :]).
    targets : np.ndarray, shape (num_samples, lead_time, num_cols).
    """
    num_timesteps, num_cols = series.shape
    num_samples = num_timesteps - order - lead_time + 1
    if num_samples < 1:
        raise ValueError(f"series has {num_timesteps} timesteps, too short for "
                          f"order={order} + lead_time={lead_time}")
    windows = np.lib.stride_tricks.sliding_window_view(
        series, window_shape=order, axis=0)[:num_samples]  # (num_samples, num_cols, order)
    windows = np.transpose(windows, (0, 2, 1))  # (num_samples, order, num_cols)
    targets = np.stack([series[i + order: i + order + lead_time] for i in range(num_samples)], axis=0)
    return windows, targets


def forecast_coeffs(density, freqs, coeffs, seq_len, lead_time, target, eval_split='test'):
    """Roll already-fitted AR coefficients out over `eval_split`, returning
    raw physical arrays rather than a metrics dict — the array-returning
    counterpart to evaluate_coeffs, for callers (e.g.
    scripts/compare_versions.py) that want to hand the forecasts to another
    metrics function (e.g. nn/spectrum_eval.py::compute_density_metrics, for
    the 'density' target, to reuse its exact formulas) or plot them directly.

    Parameters
    ----------
    density   : pd.DataFrame (time, num_freqs), float frequency columns.
    freqs     : torch.Tensor | np.ndarray, shape (num_freqs,).
    coeffs    : np.ndarray, from fit_linear_ar/fit_linear_ar_from_density.
    seq_len   : int, AR order == lookback window used to fit `coeffs`.
    lead_time : int, forecast horizon in steps.
    target    : 'hs', 'density', or 'shape'.
    eval_split: 'val' or 'test' — which split to score against.

    Returns
    -------
    (pred, true, pers) : np.ndarray, each shape (num_samples, lead_time,
    num_cols) — num_cols is 1 for 'hs', num_freqs for 'density'/'shape'.
    """
    if target not in ('hs', 'density', 'shape'):
        raise ValueError(f"Invalid target {target!r}. Choose 'hs', 'density', or 'shape'.")
    if eval_split not in ('val', 'test'):
        raise ValueError(f"eval_split must be 'val' or 'test', got {eval_split!r}")

    freqs_np = _to_numpy(freqs)
    _, val_density, test_density = _split_train_val_test(density)
    eval_density = val_density if eval_split == 'val' else test_density
    eval_series = _derive_target_series(eval_density, freqs_np, target)

    windows, y_true_np = _make_windows(eval_series, seq_len, lead_time)
    y_pers_np = np.repeat(windows[:, -1:, :], lead_time, axis=1)  # last observed value, broadcast
    y_pred_np = _rollout(coeffs, windows, lead_time, freqs=freqs_np if target == 'shape' else None)
    return y_pred_np, y_true_np, y_pers_np


def _compute_metrics(y_pred_np, y_true_np, y_pers_np, freqs_np, target):
    """Metrics for already-forecasted physical arrays (see forecast_coeffs).

    Returns
    -------
    dict with 'per_step_RMSE', 'per_step_RMSE_pers', 'per_step_SS',
    'per_step_Bias'; plus for 'hs': 'Hs_SS', 'RMSE', 'CC', 'R2', 'overall_SS',
    'Hs_MAPE' (all pooled across steps, unweighted — 'hs' has no frequency
    axis, and these are physical, so directly comparable to a checkpoint's
    own 'hs'-target evaluate() output); plus for 'density': 'Hs_RMSE',
    'Hs_SS', 'Hs_Bias', 'Tm02_RMSE', 'Tm02_Bias', 'Shape_RMSE', 'Shape_SS',
    'Shape_masked_samples', 'SI_per_bin', 'SI_mean' (final forecast step
    only, same convention as nn/evaluate.py's density block); plus for
    'shape': 'Shape_RMSE', 'Shape_SS' (aliasing per_step_RMSE[-1]/
    per_step_SS[-1], same convention nn/evaluate.py uses). No 'Hs_SS' key
    for 'shape' — there is no magnitude information in a pure shape target,
    and nn/evaluate.py's own 'Hs_SS' for that target is an artifact (it's
    left equal to the shape-space per_step_SS[-1], never overwritten) rather
    than a deliberate metric, so it isn't reproduced here.
    """
    lead_time = y_pred_np.shape[1]
    y_pred = torch.from_numpy(y_pred_np).float()
    y_true = torch.from_numpy(y_true_np).float()
    y_pers = torch.from_numpy(y_pers_np).float()

    freq_weights = None
    if target != 'hs':
        freq_weights = torch.from_numpy(trapz_weights(freqs_np)).to(dtype=torch.float32)

    rmse_fn = RMSELoss()
    per_step_rmse, per_step_rmse_pers, per_step_ss, per_step_bias = [], [], [], []
    for step in range(lead_time):
        pred_s, true_s, pers_s = y_pred[:, step, :], y_true[:, step, :], y_pers[:, step, :]
        rmse_s = rmse_fn(pred_s, true_s, weights=freq_weights).item()
        rmse_p = rmse_fn(pers_s, true_s, weights=freq_weights).item()
        diff = pred_s - true_s
        bias_s = diff.mean().item() if freq_weights is None else (diff * freq_weights).sum(dim=-1).mean().item()
        per_step_rmse.append(rmse_s)
        per_step_rmse_pers.append(rmse_p)
        per_step_ss.append(1.0 - rmse_s / rmse_p if rmse_p > 0 else float('nan'))
        per_step_bias.append(bias_s)

    metrics = {
        'per_step_RMSE': per_step_rmse,
        'per_step_RMSE_pers': per_step_rmse_pers,
        'per_step_SS': per_step_ss,
        'per_step_Bias': per_step_bias,
    }

    if target == 'hs':
        # Pooled over every (sample, step) pair, unweighted (no frequency
        # axis) — same convention nn/evaluate.py uses for the 'hs' target,
        # and always physical on both sides (nn/evaluate.py never puts 'hs'
        # through the log/exp pipeline), so directly comparable.
        overall_rmse = rmse_fn(y_pred, y_true).item()
        overall_rmse_pers = rmse_fn(y_pers, y_true).item()
        pred_flat = y_pred_np.reshape(-1)
        true_flat = y_true_np.reshape(-1)
        cc = float(np.corrcoef(pred_flat, true_flat)[0, 1])
        ss_res = float(np.sum((true_flat - pred_flat) ** 2))
        ss_tot = float(np.sum((true_flat - true_flat.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        hs_mape = float(100.0 * np.mean(np.abs(pred_flat - true_flat) / np.abs(true_flat)))
        metrics.update({
            'Hs_SS': per_step_ss[-1],
            'RMSE': overall_rmse,
            'CC': cc,
            'R2': r2,
            'overall_SS': 1.0 - overall_rmse / overall_rmse_pers if overall_rmse_pers > 0 else float('nan'),
            'Hs_MAPE': hs_mape,
        })

    elif target == 'density':
        # Final forecast step only — same convention as nn/evaluate.py's
        # density block (intermediate autoregressive steps are scaffolding,
        # not a deliverable in their own right).
        pred_final = y_pred_np[:, -1, :]   # (num_samples, num_freqs)
        true_final = y_true_np[:, -1, :]
        pers_final = y_pers_np[:, -1, :]

        hs_pred, tm02_pred = compute_bulk_params(pred_final, freqs_np)
        hs_true, tm02_true = compute_bulk_params(true_final, freqs_np)
        hs_pers, _ = compute_bulk_params(pers_final, freqs_np)

        hs_err = hs_pred - hs_true
        tm02_err = tm02_pred - tm02_true
        hs_rmse_pers = float(np.sqrt(np.mean((hs_pers - hs_true) ** 2)))
        hs_rmse_model = float(np.sqrt(np.mean(hs_err ** 2)))

        m0_true = np.trapezoid(true_final, freqs_np, axis=1)  # (num_samples,)
        valid = m0_true >= M0_MASK_THRESHOLD
        n_masked = int((~valid).sum())
        freq_w = trapz_weights(freqs_np)

        if valid.any():
            m0_denom = np.where(valid, m0_true, 1.0)[:, np.newaxis]
            shape_pred = pred_final / m0_denom
            shape_true = true_final / m0_denom
            shape_pers = pers_final / m0_denom

            per_spectrum_rmse = np.sqrt((((shape_pred - shape_true) ** 2) * freq_w).sum(axis=1))
            per_spectrum_rmse_pers = np.sqrt((((shape_pers - shape_true) ** 2) * freq_w).sum(axis=1))
            shape_rmse = float(per_spectrum_rmse[valid].mean())
            shape_rmse_pers = float(per_spectrum_rmse_pers[valid].mean())
            shape_ss = 1.0 - shape_rmse / shape_rmse_pers if shape_rmse_pers > 0 else float('nan')
        else:
            shape_rmse = float('nan')
            shape_ss = float('nan')

        rmse_per_bin = np.sqrt(((pred_final - true_final) ** 2).mean(axis=0))
        mean_per_bin = true_final.mean(axis=0).clip(min=1e-12)
        si_per_bin = rmse_per_bin / mean_per_bin
        si_mean = float((si_per_bin * freq_w).sum())

        metrics.update({
            'Hs_RMSE': hs_rmse_model,
            'Hs_SS': 1.0 - hs_rmse_model / hs_rmse_pers if hs_rmse_pers > 0 else float('nan'),
            'Hs_Bias': float(np.mean(hs_err)),
            'Tm02_RMSE': float(np.sqrt(np.mean(tm02_err ** 2))),
            'Tm02_Bias': float(np.mean(tm02_err)),
            'Shape_RMSE': shape_rmse,
            'Shape_SS': shape_ss,
            'Shape_masked_samples': n_masked,
            'SI_per_bin': si_per_bin.tolist(),
            'SI_mean': si_mean,
        })

    elif target == 'shape':
        # Already physical (this baseline never leaves physical space) —
        # aliases the final-step per_step_RMSE/SS, same convention as
        # nn/evaluate.py's shape block.
        metrics.update({
            'Shape_RMSE': per_step_rmse[-1],
            'Shape_SS': per_step_ss[-1],
        })

    return metrics


def evaluate_coeffs(density, freqs, coeffs, seq_len, lead_time, target, eval_split='test'):
    """Score already-fitted AR coefficients (e.g. loaded from a checkpoint
    saved by scripts/train_linear_baseline.py) against `eval_split`, without
    refitting. See _compute_metrics for the returned keys."""
    freqs_np = _to_numpy(freqs)
    y_pred_np, y_true_np, y_pers_np = forecast_coeffs(
        density, freqs, coeffs, seq_len, lead_time, target, eval_split=eval_split)
    return _compute_metrics(y_pred_np, y_true_np, y_pers_np, freqs_np, target)


def evaluate_linear_ar(density, freqs, seq_len, lead_time, target, ridge=1e-6, eval_split='test'):
    """Fit a per-frequency-bin (or scalar, for 'hs') linear AR baseline on
    the train split and evaluate it on `eval_split` (default 'test'), using
    the same 70/15/15 chronological split nn/optimization.py uses, so
    results are directly comparable to a transformer experiment run with the
    same (seq_len, lead_time, target).

    Convenience wrapper combining fit_linear_ar_from_density + evaluate_coeffs
    — call those directly when scoring an already-fitted checkpoint (e.g.
    scripts/compare_versions.py) to avoid refitting.

    Parameters
    ----------
    density   : pd.DataFrame (time, num_freqs), float frequency columns — same
                convention as nn/prepare_y.py.
    freqs     : torch.Tensor | np.ndarray, shape (num_freqs,).
    seq_len   : int, AR order == lookback window (match the compared model's
                seq_len).
    lead_time : int, forecast horizon in steps.
    target    : 'hs', 'density', or 'shape' — same vocabulary as
                nn/prepare_y.py.
    ridge     : forwarded to fit_linear_ar.
    eval_split: 'val' or 'test' — which split to score against (e.g. 'val'
                when grid-searching seq_len/ridge, 'test' for a final report).

    Returns
    -------
    See _compute_metrics.
    """
    coeffs = fit_linear_ar_from_density(density, freqs, seq_len, target, ridge=ridge)
    return evaluate_coeffs(density, freqs, coeffs, seq_len, lead_time, target, eval_split=eval_split)
