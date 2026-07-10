"""
Compare forecast accuracy between two experiments on the FULL PHYSICAL DENSITY
SPECTRUM E(f) — i.e. the metric set nn/evaluate.py computes for a
'density'-target model — regardless of whether either experiment trains a
single monolithic 'density' model (e.g. weightedmeanSS_conv_freqemb_v3) or a
split 'hs' + 'shape' pair recombined at inference time (e.g. hs_shape_v5, see
CLAUDE.md's shape/magnitude model split and scripts/infer.py --target
combined). This lets you ask "is experiment B actually better at forecasting
the spectrum than experiment A?" even when B never trains a density model
directly.

For a 'combined' experiment, E_pred(f, t) = shape_pred(f, t) * m0_pred(t),
m0_pred = (Hs_pred / 4)^2, evaluated over the ENTIRE test set (not just a
handful of samples like scripts/infer.py's inspection mode).

Both experiments are evaluated with the exact same metric computation (a
local re-implementation of nn/evaluate.py's density-target block operating on
numpy arrays), so results are directly comparable regardless of which path
produced them.

Usage:
    python scripts/compare_versions.py \
        --experiment-a weightedmeanSS_conv_freqemb_v3 --target-a density \
        --experiment-b hs_shape_v5 --target-b combined \
        --lead 6
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils import get_freqs, get_start_token, compute_bulk_params, get_device
from nn import WaveHeightBaselineNN
from nn.optimization import _prepare_dataloaders
from nn.channels import CHANNEL_SETS, AUX_CHANNEL_SETS

# Categorical slots from the project's validated palette (references/palette.md
# in the dataviz skill) — slot 1 (blue) and slot 6 (red) for maximum
# separation between exactly two series; assignment is fixed across every
# plot in this script (A is always blue, B is always red).
COLOR_TRUE = '#0b0b0b'
COLOR_A = '#2a78d6'
COLOR_B = '#e34948'
COLOR_PERS = '#898781'

M0_MASK_THRESHOLD = 1e-4  # m²; matches nn/evaluate.py


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment-a', required=True)
    p.add_argument('--target-a', default='density', choices=['density', 'combined'],
                   help="'density': single density-target checkpoint. "
                        "'combined': recombine that experiment's 'hs' + 'shape' checkpoints.")
    p.add_argument('--deltat-a', type=int, default=1)
    p.add_argument('--label-a', default=None, help='Display name for experiment A (default: --experiment-a)')

    p.add_argument('--experiment-b', required=True)
    p.add_argument('--target-b', default='combined', choices=['density', 'combined'])
    p.add_argument('--deltat-b', type=int, default=1)
    p.add_argument('--label-b', default=None, help='Display name for experiment B (default: --experiment-b)')

    p.add_argument('--lead', type=int, required=True, help='Lead time in hours')
    p.add_argument('--seed', type=int, default=42, help='Seed suffix of the checkpoint(s) to load')
    p.add_argument('--channel-set', default='full', choices=list(CHANNEL_SETS))
    p.add_argument('--aux-set', default='none', choices=list(AUX_CHANNEL_SETS))
    p.add_argument('--n-example-samples', type=int, default=4,
                   help='Number of example forecast times to plot full spectra for')
    p.add_argument('--out-dir', default=None,
                   help='Where to write metrics.json and plots (default: '
                        'results/comparisons/{experiment-a}_vs_{experiment-b}/lead_{N}h)')
    return p.parse_args()


def find_checkpoint(project_root, experiment, target, deltat, lead, seed):
    """Same lookup convention as scripts/infer.py's find_checkpoint."""
    candidates = [
        project_root / 'results' / experiment / target / f'deltat_{deltat}' / f'lead_{lead}h' / f'final_model_seed{seed}.pt',
        project_root / 'results' / experiment / target / f'lead_{lead}h' / f'final_model_seed{seed}.pt',
    ]
    for c in candidates:
        if c.exists():
            print(f"  loaded {target!r} checkpoint: {c}")
            return torch.load(c, map_location='cpu', weights_only=False)
    raise FileNotFoundError(f"No checkpoint for experiment={experiment!r} target={target!r} "
                            f"deltat={deltat} lead={lead}h seed={seed}. Looked in {candidates}.")


def build_model(ckpt, freqs, device, channel_set, aux_set):
    p = ckpt['params']
    model = WaveHeightBaselineNN(
        num_freqs=len(freqs), freqs=freqs, target=ckpt['target'],
        num_channels=len(CHANNEL_SETS[channel_set]),
        num_aux_channels=len(AUX_CHANNEL_SETS[aux_set]),
        dropout=p['dropout'], nhead=p['nhead'],
        num_encoder_layers=p['num_encoder_layers'],
        num_decoder_layers=p['num_decoder_layers'],
        embed_dim=p['head_dim'] * p['nhead'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model


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
    hs_ckpt = find_checkpoint(project_root, experiment, 'hs', deltat, lead, seed)
    shape_ckpt = find_checkpoint(project_root, experiment, 'shape', deltat, lead, seed)
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
    hs_err, tm02_err = hs_pred - hs_true, tm02_pred - tm02_true
    hs_mape = float(100.0 * np.mean(np.abs(hs_err) / np.abs(hs_true)))

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
        'Tm02_RMSE': float(np.sqrt(np.mean(tm02_err ** 2))), 'Tm02_Bias': float(np.mean(tm02_err)),
        'Shape_RMSE': shape_rmse, 'Shape_masked_samples': n_masked,
        'SI_per_bin': si_per_bin.tolist(), 'SI_mean': float(si_per_bin.mean()),
    }


SUMMARY_METRICS = [
    ('RMSE', 'm²/Hz'), ('R2', ''), ('CC', ''),
    ('Hs_RMSE', 'm'), ('Tm02_RMSE', 's'), ('Shape_RMSE', ''), ('SI_mean', ''),
]


def print_comparison(label_a, label_b, metrics_a, metrics_b):
    print(f"\n{'Metric':<16}{label_a:>18}{label_b:>18}{'Δ (B-A)':>14}")
    print('-' * 66)
    for key, unit in SUMMARY_METRICS:
        a, b = metrics_a[key], metrics_b[key]
        delta = b - a
        print(f"{key:<16}{a:>15.4f} {unit:<2}{b:>15.4f} {unit:<2}{delta:>+14.4f}")
    print(f"{'overall_SS':<16}{metrics_a['overall_SS']:>18.4f}{metrics_b['overall_SS']:>18.4f}"
          f"{metrics_b['overall_SS'] - metrics_a['overall_SS']:>+14.4f}")
    print(f"{'n_samples':<16}{metrics_a['n_samples']:>18d}{metrics_b['n_samples']:>18d}")


def plot_summary_bars(metrics_a, metrics_b, label_a, label_b, out_path):
    """Small multiples — one subplot per metric — since these metrics span
    different units/scales and must never share a y-axis (see dataviz skill's
    'one axis' rule)."""
    fig, axes = plt.subplots(1, len(SUMMARY_METRICS), figsize=(3 * len(SUMMARY_METRICS), 3.2))
    for ax, (key, unit) in zip(axes, SUMMARY_METRICS):
        vals = [metrics_a[key], metrics_b[key]]
        ax.bar([0, 1], vals, color=[COLOR_A, COLOR_B], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label_a, label_b], rotation=20, ha='right', fontsize=8)
        ax.set_title(f"{key}" + (f" ({unit})" if unit else ''), fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.suptitle(f"{label_a} vs {label_b} — full density spectrum, test set")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_per_step(metrics_a, metrics_b, label_a, label_b, out_path):
    lead_steps = len(metrics_a['per_step_RMSE'])
    x = np.arange(1, lead_steps + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(x, metrics_a['per_step_RMSE'], '-o', color=COLOR_A, label=label_a, linewidth=2, markersize=5)
    ax.plot(x, metrics_b['per_step_RMSE'], '-o', color=COLOR_B, label=label_b, linewidth=2, markersize=5)
    ax.plot(x, metrics_a['per_step_RMSE_pers'], '--', color=COLOR_PERS, label='Persistence', linewidth=1.5)
    ax.set_xlabel('Forecast step (hours ahead)')
    ax.set_ylabel('RMSE (m²/Hz)')
    ax.set_title('Per-step spectrum RMSE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x, metrics_a['per_step_SS'], '-o', color=COLOR_A, label=label_a, linewidth=2, markersize=5)
    ax.plot(x, metrics_b['per_step_SS'], '-o', color=COLOR_B, label=label_b, linewidth=2, markersize=5)
    ax.axhline(0.0, color=COLOR_PERS, linestyle='--', linewidth=1.5, label='Persistence (SS=0)')
    ax.set_xlabel('Forecast step (hours ahead)')
    ax.set_ylabel('Skill Score')
    ax.set_title('Per-step Skill Score vs persistence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_si_per_bin(metrics_a, metrics_b, freqs_np, label_a, label_b, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(freqs_np, metrics_a['SI_per_bin'], '-', color=COLOR_A, label=label_a, linewidth=2)
    ax.plot(freqs_np, metrics_b['SI_per_bin'], '-', color=COLOR_B, label=label_b, linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz, log scale)')
    ax.set_ylabel('Scatter Index (RMSE / mean)')
    ax.set_title('Per-frequency-bin Scatter Index')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_example_spectra(true_common, pers_common, pred_a_common, pred_b_common,
                         t0_examples, freqs_np, label_a, label_b, lead_time_steps, out_path):
    """Full spectrum at the FINAL forecast step (i.e. the actual --lead horizon)
    for a handful of example forecast-start times, drawn from the t0 range
    common to both experiments so True/Persistence are identical references."""
    n = len(t0_examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    last_step = lead_time_steps - 1
    for ax, i, t0 in zip(axes, range(n), t0_examples):
        ax.plot(freqs_np, true_common[i, last_step], '-', color=COLOR_TRUE, label='True', linewidth=2)
        ax.plot(freqs_np, pred_a_common[i, last_step], '-', color=COLOR_A, label=label_a, linewidth=1.75)
        ax.plot(freqs_np, pred_b_common[i, last_step], '-', color=COLOR_B, label=label_b, linewidth=1.75)
        ax.plot(freqs_np, pers_common[i, last_step], '--', color=COLOR_PERS, label='Persistence', linewidth=1.25)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('E(f) (m²/Hz)')
        ax.set_title(f't0={t0} (+{lead_time_steps}h)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"{label_a} vs {label_b} — example reconstructed spectra at the {lead_time_steps}h horizon")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    label_a = args.label_a or args.experiment_a
    label_b = args.label_b or args.experiment_b
    project_root = Path(__file__).resolve().parent.parent
    device = get_device()
    print(f"Device: {device}")

    density, alpha_1, alpha_2, r_1, wind = pd.read_pickle(project_root / 'buoy_data' / 'processed_data.pkl')
    freqs = get_freqs(density)
    freqs_np = freqs.cpu().numpy() if torch.is_tensor(freqs) else np.asarray(freqs)

    def load(experiment, target, deltat):
        density_d, alpha_1_d, alpha_2_d, r_1_d, wind_d = (
            density[::deltat], alpha_1[::deltat], alpha_2[::deltat], r_1[::deltat], wind[::deltat])
        if target == 'density':
            ckpt = find_checkpoint(project_root, experiment, 'density', deltat, args.lead, args.seed)
            return eval_single_density(ckpt, density_d, alpha_1_d, alpha_2_d, r_1_d, wind_d,
                                       freqs, device, args.channel_set, args.aux_set, args.seed)
        else:
            return eval_combined(project_root, experiment, deltat, args.lead, args.seed,
                                 density_d, alpha_1_d, alpha_2_d, r_1_d, wind_d,
                                 freqs, device, args.channel_set, args.aux_set)

    print(f"\nEvaluating A: {label_a} (target={args.target_a})")
    pred_a, true_a, pers_a, lead_a, t0_start_a = load(args.experiment_a, args.target_a, args.deltat_a)
    print(f"Evaluating B: {label_b} (target={args.target_b})")
    pred_b, true_b, pers_b, lead_b, t0_start_b = load(args.experiment_b, args.target_b, args.deltat_b)

    if lead_a != lead_b:
        raise ValueError(f"lead_time_steps mismatch: A={lead_a} B={lead_b} — "
                         "comparison requires the same forecast horizon.")
    lead_time_steps = lead_a

    metrics_a = compute_density_metrics(pred_a, true_a, pers_a, freqs_np)
    metrics_b = compute_density_metrics(pred_b, true_b, pers_b, freqs_np)
    print_comparison(label_a, label_b, metrics_a, metrics_b)

    out_dir = Path(args.out_dir) if args.out_dir else (
        project_root / 'results' / 'comparisons' / f'{args.experiment_a}_vs_{args.experiment_b}' / f'lead_{args.lead}h')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'label_a': label_a, 'label_b': label_b,
                   'target_a': args.target_a, 'target_b': args.target_b,
                   'metrics_a': metrics_a, 'metrics_b': metrics_b}, f, indent=2)
    print(f"\nSaved metrics.json to {out_dir}")

    print("\nGenerating plots...")
    plot_summary_bars(metrics_a, metrics_b, label_a, label_b, out_dir / 'summary_bars.png')
    plot_per_step(metrics_a, metrics_b, label_a, label_b, out_dir / 'per_step.png')
    plot_si_per_bin(metrics_a, metrics_b, freqs_np, label_a, label_b, out_dir / 'si_per_bin.png')

    # Example spectra: only t0 values covered by BOTH experiments' encoder
    # windows have directly comparable True/Persistence references.
    end_a, end_b = t0_start_a + pred_a.shape[0] - 1, t0_start_b + pred_b.shape[0] - 1
    common_start, common_end = max(t0_start_a, t0_start_b), min(end_a, end_b)
    if common_end >= common_start:
        n_examples = min(args.n_example_samples, common_end - common_start + 1)
        t0_examples = np.linspace(common_start, common_end, n_examples, dtype=int)
        idx_a = t0_examples - t0_start_a
        idx_b = t0_examples - t0_start_b
        plot_example_spectra(true_a[idx_a], pers_a[idx_a], pred_a[idx_a], pred_b[idx_b],
                             t0_examples, freqs_np, label_a, label_b, lead_time_steps,
                             out_dir / 'example_spectra.png')
    else:
        print("  no overlapping forecast-start range between A and B — skipping example_spectra.png")

    print(f"\nDone. All outputs under {out_dir}")


if __name__ == '__main__':
    main()
