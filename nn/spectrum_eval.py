"""Full-test-set metric computation for a physical density spectrum E(f) —
usable both for a monolithic 'density'-target checkpoint and for a
recombined 'hs'+'shape' checkpoint pair (see CLAUDE.md's shape/magnitude
model split, scripts/infer.py's --target combined, and
scripts/compare_versions.py). Extracted so infer.py (single-experiment
metrics) and compare_versions.py (multi-experiment comparison) share one
implementation instead of drifting apart.
"""
import numpy as np
import torch

from utils import get_start_token, compute_bulk_params
from nn.checkpoints import find_checkpoint, build_model
from nn.optimization import _prepare_dataloaders

M0_MASK_THRESHOLD = 1e-4  # m²; matches nn/evaluate.py


def eval_single_density(ckpt, density_d, alpha_1_d, alpha_2_d, r_1_d, wind_d,
                        freqs, device, channel_set, aux_set, seed):
    """Autoregressive inference for a monolithic density-target checkpoint.

    Returns physical (pred, true, pers) arrays of shape (N, lead_time, num_freqs)
    plus lead_time_steps and t0_start — the test-split-relative row index of
    the first forecast step for sample 0 (samples are contiguous: sample j's
    forecast starts at t0_start + j, since the test DataLoader is unshuffled).
    """
    params = ckpt['params']
    model = build_model(ckpt, freqs, device, channel_set, aux_set)
    freq_means = ckpt['freq_means'].to(device)
    fm_np = freq_means.cpu().numpy()
    lead_time_steps = ckpt['lead_time_steps']

    _, _, test_loader, _, _, _, _ = _prepare_dataloaders(
        density_d, alpha_1_d, alpha_2_d, r_1_d, params['seq_len'], lead_time_steps,
        params['batch_size'], 'density', shuffle_seed=seed, wind=wind_d,
        channel_set=channel_set, aux_set=aux_set)

    all_pred, all_true, all_pers = [], [], []
    with torch.no_grad():
        for X, aux, y in test_loader:
            X, aux, y = X.to(device), aux.to(device), y.to(device)
            start_token = get_start_token(X, 'density', freqs, device, freq_means=freq_means)
            pers = start_token.unsqueeze(1).expand(-1, y.shape[1], -1)
            pred = model.infer(X, freqs, lead_time_steps, freq_means=freq_means, aux=aux)

            all_pred.append(pred.cpu().numpy() * fm_np)
            all_true.append(y.cpu().numpy() * fm_np)
            all_pers.append(pers.cpu().numpy() * fm_np)

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    pers_np = np.concatenate(all_pers, axis=0)
    t0_start = params['seq_len']
    return pred_np, true_np, pers_np, lead_time_steps, t0_start


def eval_combined(project_root, experiment, deltat, lead, seed, density_d, alpha_1_d,
                  alpha_2_d, r_1_d, wind_d, freqs, device, channel_set, aux_set):
    """Recombine an experiment's separately-trained 'hs' and 'shape' checkpoints
    into a full physical density spectrum, evaluated over the entire test set
    (see scripts/infer.py's --target combined, which does this for a handful
    of inspection samples only).

    Returns the same (pred, true, pers, lead_time_steps, t0_start) shape as
    eval_single_density, so downstream code is agnostic to which path produced it.
    """
    hs_ckpt, _ = find_checkpoint(project_root, experiment, 'hs', deltat, lead, seed)
    shape_ckpt, _ = find_checkpoint(project_root, experiment, 'shape', deltat, lead, seed)
    lead_time_steps = hs_ckpt['lead_time_steps']
    if shape_ckpt['lead_time_steps'] != lead_time_steps:
        raise ValueError(f"hs lead_time_steps={lead_time_steps} != "
                         f"shape lead_time_steps={shape_ckpt['lead_time_steps']}")

    hs_model = build_model(hs_ckpt, freqs, device, channel_set, aux_set)
    shape_model = build_model(shape_ckpt, freqs, device, channel_set, aux_set)
    hs_freq_means = hs_ckpt['freq_means'].to(device)
    shape_freq_means = shape_ckpt['freq_means'].to(device)
    hs_params = hs_ckpt['params']
    shape_params = shape_ckpt['params']

    _, _, hs_test_loader, _, _, _, _ = _prepare_dataloaders(
        density_d, alpha_1_d, alpha_2_d, r_1_d, hs_params['seq_len'], lead_time_steps,
        hs_params['batch_size'], 'hs', shuffle_seed=seed, wind=wind_d,
        channel_set=channel_set, aux_set=aux_set)
    _, _, shape_test_loader, _, _, _, _ = _prepare_dataloaders(
        density_d, alpha_1_d, alpha_2_d, r_1_d, shape_params['seq_len'], lead_time_steps,
        shape_params['batch_size'], 'shape', shuffle_seed=seed, wind=wind_d,
        channel_set=channel_set, aux_set=aux_set)
    hs_dataset = hs_test_loader.dataset
    shape_dataset = shape_test_loader.dataset

    n = len(density_d)
    val_end = int(0.85 * n)
    test_density_vals = density_d[val_end:].values

    # Each sub-model was tuned with its own seq_len, so their test datasets
    # are windowed differently; only forecast-start points t0 covered by BOTH
    # models' encoder windows can be recombined (see scripts/infer.py's
    # run_combined for the same alignment logic, applied there to a handful
    # of samples instead of the whole test set).
    t0_start = max(hs_params['seq_len'], shape_params['seq_len'])
    t0_end = len(test_density_vals) - lead_time_steps - 1
    t0_values = list(range(t0_start, t0_end + 1))

    all_pred, all_true, all_pers = [], [], []
    BATCH = 128
    with torch.no_grad():
        for i in range(0, len(t0_values), BATCH):
            chunk = t0_values[i:i + BATCH]
            idx_hs = [t0 - hs_params['seq_len'] for t0 in chunk]
            idx_shape = [t0 - shape_params['seq_len'] for t0 in chunk]

            X_hs = hs_dataset.X[idx_hs].to(device)
            aux_hs = hs_dataset.aux[idx_hs].to(device)
            X_shape = shape_dataset.X[idx_shape].to(device)
            aux_shape = shape_dataset.aux[idx_shape].to(device)

            hs_pred = hs_model.infer(X_hs, freqs, lead_time_steps, freq_means=hs_freq_means, aux=aux_hs)
            shape_pred = shape_model.infer(X_shape, freqs, lead_time_steps, freq_means=shape_freq_means, aux=aux_shape)

            shape_pred_np = shape_pred.cpu().numpy()
            m0_pred = ((hs_pred / 4.0) ** 2).cpu().numpy()
            pred_phys = shape_pred_np * m0_pred

            true_phys = np.stack([test_density_vals[t0:t0 + lead_time_steps] for t0 in chunk])
            pers_phys = np.stack([np.tile(test_density_vals[t0 - 1], (lead_time_steps, 1)) for t0 in chunk])

            all_pred.append(pred_phys)
            all_true.append(true_phys)
            all_pers.append(pers_phys)

    pred_np = np.concatenate(all_pred, axis=0)
    true_np = np.concatenate(all_true, axis=0)
    pers_np = np.concatenate(all_pers, axis=0)
    return pred_np, true_np, pers_np, lead_time_steps, t0_start


def compute_density_metrics(pred_np, true_np, pers_np, freqs_np):
    """Mirrors nn/evaluate.py's overall + density-target-only metric block,
    operating on physical numpy arrays of shape (N, lead_time, num_freqs)
    instead of a live model + dataloader, so it applies identically whether
    the arrays came from a single density model or a recombined hs+shape pair.
    """
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    per_step_rmse, per_step_rmse_pers, per_step_ss, per_step_bias, per_step_r2 = [], [], [], [], []
    for step in range(pred_np.shape[1]):
        p, t, pe = pred_np[:, step, :].flatten(), true_np[:, step, :].flatten(), pers_np[:, step, :].flatten()
        r_s, r_p = rmse(p, t), rmse(pe, t)
        per_step_rmse.append(r_s)
        per_step_rmse_pers.append(r_p)
        per_step_ss.append(1.0 - r_s / r_p if r_p > 0 else float('nan'))
        per_step_bias.append(float(np.mean(p - t)))
        ss_res, ss_tot = np.sum((t - p) ** 2), np.sum((t - t.mean()) ** 2)
        per_step_r2.append(float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan'))

    pred_flat, true_flat, pers_flat = pred_np.flatten(), true_np.flatten(), pers_np.flatten()
    overall_rmse = rmse(pred_flat, true_flat)
    overall_rmse_pers = rmse(pers_flat, true_flat)
    overall_ss = 1.0 - overall_rmse / overall_rmse_pers if overall_rmse_pers > 0 else float('nan')
    cc = float(np.corrcoef(pred_flat, true_flat)[0, 1])
    bias = float(np.mean(pred_flat - true_flat))
    ss_res, ss_tot = np.sum((true_flat - pred_flat) ** 2), np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

    hs_pred, tm02_pred = compute_bulk_params(pred_np, freqs_np)
    hs_true, tm02_true = compute_bulk_params(true_np, freqs_np)
    hs_pers, _ = compute_bulk_params(pers_np, freqs_np)
    hs_err, tm02_err = hs_pred - hs_true, tm02_pred - tm02_true
    hs_err_pers = hs_pers - hs_true
    hs_mape = float(100.0 * np.mean(np.abs(hs_err) / np.abs(hs_true)))

    # hs_err/hs_err_pers are (N, lead_time) — collapsing over axis=0 instead of
    # flattening gives the per-forecast-step Hs RMSE alongside the overall one.
    per_step_hs_rmse = np.sqrt((hs_err ** 2).mean(axis=0)).tolist()
    per_step_hs_rmse_pers = np.sqrt((hs_err_pers ** 2).mean(axis=0)).tolist()

    m0_true = np.trapezoid(true_np, freqs_np, axis=2)
    valid = m0_true >= M0_MASK_THRESHOLD
    n_masked = int((~valid).sum())
    m0_denom = np.where(valid, m0_true, 1.0)[:, :, np.newaxis]
    shape_pred_norm, shape_true_norm = pred_np / m0_denom, true_np / m0_denom
    per_spectrum_rmse = np.sqrt(((shape_pred_norm - shape_true_norm) ** 2).mean(axis=2))
    shape_rmse = float(per_spectrum_rmse[valid].mean()) if valid.any() else float('nan')

    flat_pred = pred_np.reshape(-1, pred_np.shape[2])
    flat_true = true_np.reshape(-1, true_np.shape[2])
    rmse_per_bin = np.sqrt(((flat_pred - flat_true) ** 2).mean(axis=0))
    mean_per_bin = flat_true.mean(axis=0).clip(min=1e-12)
    si_per_bin = rmse_per_bin / mean_per_bin

    return {
        'n_samples': int(pred_np.shape[0]),
        'RMSE': overall_rmse, 'CC': cc, 'Bias': bias, 'R2': r2,
        'per_step_RMSE': per_step_rmse, 'per_step_RMSE_pers': per_step_rmse_pers,
        'per_step_SS': per_step_ss, 'per_step_Bias': per_step_bias, 'per_step_R2': per_step_r2,
        'overall_SS': overall_ss,
        'Hs_MAPE': hs_mape,
        'Hs_RMSE': float(np.sqrt(np.mean(hs_err ** 2))), 'Hs_Bias': float(np.mean(hs_err)),
        'per_step_Hs_RMSE': per_step_hs_rmse, 'per_step_Hs_RMSE_pers': per_step_hs_rmse_pers,
        'Tm02_RMSE': float(np.sqrt(np.mean(tm02_err ** 2))), 'Tm02_Bias': float(np.mean(tm02_err)),
        'Shape_RMSE': shape_rmse, 'Shape_masked_samples': n_masked,
        'SI_per_bin': si_per_bin.tolist(), 'SI_mean': float(si_per_bin.mean()),
    }
