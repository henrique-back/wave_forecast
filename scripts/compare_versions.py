"""
Compare forecast accuracy between two OR MORE experiments on the FULL PHYSICAL
DENSITY SPECTRUM E(f) — i.e. the metric set nn/evaluate.py computes for a
'density'-target model — regardless of whether an experiment trains a single
monolithic 'density' model (e.g. weightedmeanSS_conv_freqemb_v3) or a split
'hs' + 'shape' pair recombined at inference time (e.g. hs_shape_v5, see
CLAUDE.md's shape/magnitude model split and scripts/infer.py --target
combined). This lets you ask "is experiment B actually better at forecasting
the spectrum than experiment A?" even when B never trains a density model
directly — and extends the same question across any number of experiments.

For a 'combined' experiment, E_pred(f, t) = shape_pred(f, t) * m0_pred(t),
m0_pred = (Hs_pred / 4)^2, evaluated over the ENTIRE test set (not just a
handful of samples like scripts/infer.py's inspection mode). When at least one
compared experiment uses target=combined, per_step.png additionally breaks
out the per-step Hs RMSE (the bulk parameter driving that recombination) for
every experiment, so you can see whether spectrum-level errors trace back to
the Hs sub-model specifically.

All experiments are evaluated with the exact same metric computation (a local
re-implementation of nn/evaluate.py's density-target block operating on numpy
arrays), so results are directly comparable regardless of which path produced
them.

Usage:
    python scripts/compare_versions.py \
        --experiment weightedmeanSS_conv_freqemb_v3:density \
        --experiment hs_shape_v5:combined:"HS+Shape v5" \
        --experiment hs_shape_v6:combined:"HS+Shape v6" \
        --lead 6

Each --experiment is repeatable (2 or more required) and formatted as
NAME:TARGET[:LABEL], where TARGET is 'density' or 'combined' and LABEL
defaults to NAME if omitted.
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
# in the dataviz skill), in fixed hue order — assigned to experiments in the
# order given on the command line so identity never depends on rank.
CATEGORICAL_PALETTE = [
    '#2a78d6',  # 1 blue
    '#1baf7a',  # 2 aqua
    '#eda100',  # 3 yellow
    '#008300',  # 4 green
    '#4a3aa7',  # 5 violet
    '#e34948',  # 6 red
    '#e87ba4',  # 7 magenta
    '#eb6834',  # 8 orange
]
COLOR_TRUE = '#0b0b0b'
COLOR_PERS = '#898781'

M0_MASK_THRESHOLD = 1e-4  # m²; matches nn/evaluate.py


def parse_experiment_spec(s):
    parts = s.split(':')
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"--experiment must be formatted NAME:TARGET[:LABEL], got {s!r}")
    name, target = parts[0], parts[1]
    label = parts[2] if len(parts) == 3 else name
    if target not in ('density', 'combined'):
        raise argparse.ArgumentTypeError(
            f"target must be 'density' or 'combined', got {target!r} (in {s!r})")
    return {'name': name, 'target': target, 'label': label}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', dest='experiments', action='append', required=True,
                   type=parse_experiment_spec, metavar='NAME:TARGET[:LABEL]',
                   help="Repeatable, at least 2 required. TARGET is 'density' "
                        "('density'-target checkpoint) or 'combined' (recombine "
                        "that experiment's 'hs' + 'shape' checkpoints). LABEL "
                        "is the display name, defaults to NAME.")

    p.add_argument('--lead', type=int, required=True, help='Lead time in hours')
    p.add_argument('--seed', type=int, default=42, help='Seed suffix of the checkpoint(s) to load')
    p.add_argument('--channel-set', default='full', choices=list(CHANNEL_SETS))
    p.add_argument('--aux-set', default='none', choices=list(AUX_CHANNEL_SETS))
    p.add_argument('--n-example-samples', type=int, default=4,
                   help='Number of example forecast times to plot full spectra for')
    p.add_argument('--out-dir', default=None,
                   help='Where to write metrics.json and plots (default: '
                        'results/comparisons/{name1}_vs_{name2}_vs_.../lead_{N}h)')
    args = p.parse_args()
    if len(args.experiments) < 2:
        p.error('at least 2 --experiment entries are required for a comparison')
    return args


def find_checkpoint(project_root, experiment, target, lead, seed):
    """Same lookup convention as scripts/infer.py's find_checkpoint."""
    candidates = [
        project_root / 'results' / experiment / target / f'lead_{lead}h' / f'final_model_seed{seed}.pt',
        project_root / 'results' / experiment / target / f'lead_{lead}h' / f'final_model_seed{seed}.pt',
    ]
    for c in candidates:
        if c.exists():
            print(f"  loaded {target!r} checkpoint: {c}")
            return torch.load(c, map_location='cpu', weights_only=False)
    raise FileNotFoundError(f"No checkpoint for experiment={experiment!r} target={target!r} "
                            f"lead={lead}h seed={seed}. Looked in {candidates}.")


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


def eval_combined(project_root, experiment, lead, seed, density_d, alpha_1_d,
                  alpha_2_d, r_1_d, wind_d, freqs, device, channel_set, aux_set):
    """Recombine an experiment's separately-trained 'hs' and 'shape' checkpoints
    into a full physical density spectrum, evaluated over the entire test set
    (see scripts/infer.py's --target combined, which does this for a handful
    of inspection samples only).

    Returns the same (pred, true, pers, lead_time_steps, t0_start) shape as
    eval_single_density, so downstream code is agnostic to which path produced it.
    """
    hs_ckpt = find_checkpoint(project_root, experiment, 'hs', lead, seed)
    shape_ckpt = find_checkpoint(project_root, experiment, 'shape', lead, seed)
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


SUMMARY_METRICS = [
    ('RMSE', 'm²/Hz'), ('R2', ''), ('CC', ''),
    ('Hs_RMSE', 'm'), ('Tm02_RMSE', 's'), ('Shape_RMSE', ''), ('SI_mean', ''),
]


def print_comparison(results):
    labels = [r['label'] for r in results]
    col_w = 16
    print(f"\n{'Metric':<{col_w}}" + ''.join(f"{l:>18}" for l in labels))
    print('-' * (col_w + 18 * len(labels)))
    for key, unit in SUMMARY_METRICS:
        row = f"{key:<{col_w}}"
        for r in results:
            row += f"{r['metrics'][key]:>15.4f} {unit:<2}"
        print(row)
    print(f"{'overall_SS':<{col_w}}" + ''.join(f"{r['metrics']['overall_SS']:>18.4f}" for r in results))
    print(f"{'n_samples':<{col_w}}" + ''.join(f"{r['metrics']['n_samples']:>18d}" for r in results))

    if len(results) > 1:
        base = results[0]
        others = results[1:]
        print(f"\nΔ vs {base['label']} (per-metric, later experiment minus baseline):")
        print(f"{'Metric':<{col_w}}" + ''.join(f"{r['label']:>18}" for r in others))
        for key, unit in SUMMARY_METRICS:
            row = f"{key:<{col_w}}"
            for r in others:
                row += f"{(r['metrics'][key] - base['metrics'][key]):>+15.4f} {unit:<2}"
            print(row)


def plot_summary_bars(results, out_path):
    """Small multiples — one subplot per metric — since these metrics span
    different units/scales and must never share a y-axis (see dataviz skill's
    'one axis' rule)."""
    labels = [r['label'] for r in results]
    colors = [r['color'] for r in results]
    x = np.arange(len(results))
    fig, axes = plt.subplots(1, len(SUMMARY_METRICS), figsize=(3 * len(SUMMARY_METRICS), 3.2))
    for ax, (key, unit) in zip(axes, SUMMARY_METRICS):
        vals = [r['metrics'][key] for r in results]
        ax.bar(x, vals, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
        ax.set_title(f"{key}" + (f" ({unit})" if unit else ''), fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
    fig.suptitle(' vs '.join(labels) + ' — full density spectrum, test set')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_per_step(results, out_path):
    lead_steps = len(results[0]['metrics']['per_step_RMSE'])
    x = np.arange(1, lead_steps + 1)
    # The Hs-RMSE panel is only meaningful when at least one compared
    # experiment recombines a split hs+shape pair (see module docstring).
    show_hs_panel = any(r['target'] == 'combined' for r in results)
    n_panels = 3 if show_hs_panel else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4))

    ax = axes[0]
    for r in results:
        ax.plot(x, r['metrics']['per_step_RMSE'], '-o', color=r['color'], label=r['label'], linewidth=2, markersize=5)
    ax.plot(x, results[0]['metrics']['per_step_RMSE_pers'], '--', color=COLOR_PERS, label='Persistence', linewidth=1.5)
    ax.set_xlabel('Forecast step (hours ahead)')
    ax.set_ylabel('RMSE (m²/Hz)')
    ax.set_title('Per-step spectrum RMSE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for r in results:
        ax.plot(x, r['metrics']['per_step_SS'], '-o', color=r['color'], label=r['label'], linewidth=2, markersize=5)
    ax.axhline(0.0, color=COLOR_PERS, linestyle='--', linewidth=1.5, label='Persistence (SS=0)')
    ax.set_xlabel('Forecast step (hours ahead)')
    ax.set_ylabel('Skill Score')
    ax.set_title('Per-step Skill Score vs persistence')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    if show_hs_panel:
        ax = axes[2]
        for r in results:
            ax.plot(x, r['metrics']['per_step_Hs_RMSE'], '-o', color=r['color'], label=r['label'], linewidth=2, markersize=5)
        ax.plot(x, results[0]['metrics']['per_step_Hs_RMSE_pers'], '--', color=COLOR_PERS, label='Persistence', linewidth=1.5)
        ax.set_xlabel('Forecast step (hours ahead)')
        ax.set_ylabel('Hs RMSE (m)')
        ax.set_title('Per-step Hs RMSE (bulk parameter)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_si_per_bin(results, freqs_np, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in results:
        ax.plot(freqs_np, r['metrics']['SI_per_bin'], '-', color=r['color'], label=r['label'], linewidth=2)
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


def plot_example_spectra(results, t0_examples, freqs_np, lead_time_steps, out_path):
    """Full spectrum at the FINAL forecast step (i.e. the actual --lead horizon)
    for a handful of example forecast-start times, drawn from the t0 range
    common to every compared experiment so True/Persistence are identical
    references. Expects each result dict to carry 'true_common', 'pers_common',
    'pred_common' arrays already sliced down to t0_examples (see main())."""
    n = len(t0_examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    last_step = lead_time_steps - 1
    ref = results[0]
    for ax, i, t0 in zip(axes, range(n), t0_examples):
        ax.plot(freqs_np, ref['true_common'][i, last_step], '-', color=COLOR_TRUE, label='True', linewidth=2)
        for r in results:
            ax.plot(freqs_np, r['pred_common'][i, last_step], '-', color=r['color'], label=r['label'], linewidth=1.75)
        ax.plot(freqs_np, ref['pers_common'][i, last_step], '--', color=COLOR_PERS, label='Persistence', linewidth=1.25)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('E(f) (m²/Hz)')
        ax.set_title(f't0={t0} (+{lead_time_steps}h)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    labels = [r['label'] for r in results]
    fig.suptitle(' vs '.join(labels) + f' — example reconstructed spectra at the {lead_time_steps}h horizon')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    device = get_device()
    print(f"Device: {device}")

    if len(args.experiments) > len(CATEGORICAL_PALETTE):
        print(f"  warning: {len(args.experiments)} experiments requested but only "
              f"{len(CATEGORICAL_PALETTE)} categorical colors are defined — colors will repeat.")
    for i, spec in enumerate(args.experiments):
        spec['color'] = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]

    density, alpha_1, alpha_2, r_1, wind = pd.read_pickle(project_root / 'buoy_data' / '42056' / 'processed_data.pkl')
    freqs = get_freqs(density)
    freqs_np = freqs.cpu().numpy() if torch.is_tensor(freqs) else np.asarray(freqs)

    def load(spec):
        if spec['target'] == 'density':
            ckpt = find_checkpoint(project_root, spec['name'], 'density', args.lead, args.seed)
            return eval_single_density(ckpt, density, alpha_1, alpha_2, r_1, wind,
                                       freqs, device, args.channel_set, args.aux_set, args.seed)
        else:
            return eval_combined(project_root, spec['name'], args.lead, args.seed,
                                 density, alpha_1, alpha_2, r_1, wind,
                                 freqs, device, args.channel_set, args.aux_set)

    results = []
    for spec in args.experiments:
        print(f"\nEvaluating {spec['label']} (target={spec['target']})")
        pred, true, pers, lead_time_steps, t0_start = load(spec)
        results.append({**spec, 'pred': pred, 'true': true, 'pers': pers,
                        'lead_time_steps': lead_time_steps, 't0_start': t0_start})

    lead_time_steps = results[0]['lead_time_steps']
    for r in results[1:]:
        if r['lead_time_steps'] != lead_time_steps:
            raise ValueError(f"lead_time_steps mismatch: {results[0]['label']}={lead_time_steps} "
                             f"{r['label']}={r['lead_time_steps']} — comparison requires the same forecast horizon.")

    for r in results:
        r['metrics'] = compute_density_metrics(r['pred'], r['true'], r['pers'], freqs_np)
    print_comparison(results)

    out_dir = Path(args.out_dir) if args.out_dir else (
        project_root / 'results' / 'comparisons' / '_vs_'.join(r['name'] for r in results) / f'lead_{args.lead}h')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({
            'experiments': [
                {'name': r['name'], 'label': r['label'], 'target': r['target'], 'metrics': r['metrics']}
                for r in results
            ]
        }, f, indent=2)
    print(f"\nSaved metrics.json to {out_dir}")

    print("\nGenerating plots...")
    plot_summary_bars(results, out_dir / 'summary_bars.png')
    plot_per_step(results, out_dir / 'per_step.png')
    plot_si_per_bin(results, freqs_np, out_dir / 'si_per_bin.png')

    # Example spectra: only t0 values covered by EVERY experiment's encoder
    # window have directly comparable True/Persistence references.
    starts = [r['t0_start'] for r in results]
    ends = [r['t0_start'] + r['pred'].shape[0] - 1 for r in results]
    common_start, common_end = max(starts), min(ends)
    if common_end >= common_start:
        n_examples = min(args.n_example_samples, common_end - common_start + 1)
        t0_examples = np.linspace(common_start, common_end, n_examples, dtype=int)
        for r in results:
            idx = t0_examples - r['t0_start']
            r['true_common'] = r['true'][idx]
            r['pers_common'] = r['pers'][idx]
            r['pred_common'] = r['pred'][idx]
        plot_example_spectra(results, t0_examples, freqs_np, lead_time_steps, out_dir / 'example_spectra.png')
    else:
        print("  no forecast-start range common to every experiment — skipping example_spectra.png")

    print(f"\nDone. All outputs under {out_dir}")


if __name__ == '__main__':
    main()
