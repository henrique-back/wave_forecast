"""
Compare forecast accuracy between two OR MORE experiments.

Two comparison families, chosen by the TARGET(s) given:

- SPECTRUM family (target 'density' and/or 'combined', freely mixable): full
  physical density spectrum E(f) comparison — the metric set nn/evaluate.py
  computes for a 'density'-target model — regardless of whether an experiment
  trains a single monolithic 'density' model (e.g. weightedmeanSS_conv_freqemb_v3)
  or a split 'hs' + 'shape' pair recombined at inference time (e.g. hs_shape_v5,
  see CLAUDE.md's shape/magnitude model split and scripts/infer.py --target
  combined). For a 'combined' experiment, E_pred(f, t) = shape_pred(f, t) *
  m0_pred(t), m0_pred = (Hs_pred / 4)^2. When at least one compared experiment
  uses target=combined, per_step.png additionally breaks out the per-step Hs
  RMSE (the bulk parameter driving that recombination). Produces
  summary_bars.png, per_step.png, si_per_bin.png, example_spectra.png.

- SCALAR family (target 'hs' or 'shape', all experiments must share the SAME
  one — Hs metres and shape unit-area RMSE aren't on comparable scales):
  each checkpoint's own nn/evaluate.py metrics (RMSE, R2, CC, Hs_MAPE/overall_SS
  etc.), evaluated over the full test set directly — no recombination, no
  spectrum-only plots (those require a physical E(f), which a lone hs or shape
  checkpoint doesn't produce).

All experiments in a family are evaluated with the exact same metric
computation, so results are directly comparable regardless of which path
produced them. Both families write metrics.json under the same results/comparisons/
convention.

Usage:
    python scripts/compare_versions.py \
        --experiment weightedmeanSS_conv_freqemb_v3:density \
        --experiment hs_shape_v5:combined:"HS+Shape v5" \
        --experiment hs_shape_v6:combined:"HS+Shape v6" \
        --lead 6

    python scripts/compare_versions.py \
        --experiment hs_shape_v5:hs \
        --experiment hs_shape_v6:hs \
        --lead 6

Each --experiment is repeatable (2 or more required) and formatted as
NAME:TARGET[:LABEL], where TARGET is 'density', 'combined', 'hs', or 'shape'
and LABEL defaults to NAME if omitted.

The name 'linear_baseline' is special: instead of a transformer checkpoint
(scripts/train.py), it loads the simple linear AR baseline
(utils/linear_baseline.py) checkpoint saved by scripts/train_linear_baseline.py
for the given TARGET/--lead (no --seed — that baseline is deterministic), and
folds it in as one more entry, e.g.:

    python scripts/compare_versions.py \
        --experiment shape_v11:shape --experiment shape_v12:shape \
        --experiment linear_baseline:shape --lead 24

'combined' is not a valid TARGET for linear_baseline (no hs+shape
recombination story for it). For target 'shape' specifically, only
Shape_RMSE/Shape_SS are shown in the scalar-family table (see
SUMMARY_METRICS_SCALAR) rather than the usual RMSE/R2/CC — those are computed
in log-space for a transformer's 'shape' target (see nn/evaluate.py's
docstring) and would silently misrepresent a comparison against
linear_baseline's always-physical numbers.
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

from utils import get_freqs, get_device
from utils.linear_baseline import forecast_coeffs, evaluate_coeffs
from nn import evaluate
from nn.checkpoints import find_checkpoint, build_model
from nn.spectrum_eval import eval_single_density, eval_combined, compute_density_metrics
from nn.optimization import _prepare_dataloaders
from nn.channels import CHANNEL_SETS, AUX_CHANNEL_SETS

SPECTRUM_TARGETS = {'density', 'combined'}
SCALAR_TARGETS = {'hs', 'shape'}

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
# Distinct from both COLOR_PERS (gray) and every CATEGORICAL_PALETTE hue, so
# an --experiment linear_baseline:... entry is never visually confused with
# a transformer --experiment or with persistence.
COLOR_LINEAR = '#a05a2c'


def parse_experiment_spec(s):
    parts = s.split(':')
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"--experiment must be formatted NAME:TARGET[:LABEL], got {s!r}")
    name, target = parts[0], parts[1]
    label = parts[2] if len(parts) == 3 else name
    if target not in SPECTRUM_TARGETS | SCALAR_TARGETS:
        raise argparse.ArgumentTypeError(
            f"target must be one of {sorted(SPECTRUM_TARGETS | SCALAR_TARGETS)}, "
            f"got {target!r} (in {s!r})")
    if name == 'linear_baseline' and target == 'combined':
        raise argparse.ArgumentTypeError(
            "linear_baseline has no 'combined' variant (no hs+shape recombination "
            "story for it) — use 'density', 'hs', or 'shape'.")
    return {'name': name, 'target': target, 'label': label}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', dest='experiments', action='append', required=True,
                   type=parse_experiment_spec, metavar='NAME:TARGET[:LABEL]',
                   help="Repeatable, at least 2 required. TARGET is 'density' "
                        "('density'-target checkpoint), 'combined' (recombine "
                        "that experiment's 'hs' + 'shape' checkpoints), or a lone "
                        "'hs'/'shape' checkpoint compared on its own metrics "
                        "(density/combined are freely mixable with each other; "
                        "hs/shape must all match — see module docstring). LABEL "
                        "is the display name, defaults to NAME.")

    p.add_argument('--lead', type=int, required=True, help='Lead time in hours')
    p.add_argument('--seed', type=int, default=42, help='Seed suffix of the checkpoint(s) to load')
    p.add_argument('--channel-set', default='full', choices=list(CHANNEL_SETS))
    p.add_argument('--aux-set', default='none', choices=list(AUX_CHANNEL_SETS))
    p.add_argument('--n-example-samples', type=int, default=4,
                   help='Number of example forecast times to plot full spectra '
                        '(spectrum family) or unit-area shape curves (scalar '
                        "family, target='shape' only) for")
    p.add_argument('--out-dir', default=None,
                   help='Where to write metrics.json and plots (default: '
                        'results/comparisons/{name1}_vs_{name2}_vs_.../lead_{N}h)')
    args = p.parse_args()
    if len(args.experiments) < 2:
        p.error('at least 2 --experiment entries are required for a comparison')

    targets = {spec['target'] for spec in args.experiments}
    if not (targets <= SPECTRUM_TARGETS or len(targets) == 1):
        p.error(f"cannot mix targets {sorted(targets)} — density/combined are freely "
                f"mixable with each other, but hs/shape must all match (Hs metres and "
                f"shape unit-area RMSE aren't on comparable scales)")
    return args


SUMMARY_METRICS = [
    ('RMSE', 'm²/Hz'), ('R2', ''), ('CC', ''),
    ('Hs_RMSE', 'm'), ('Tm02_RMSE', 's'), ('Shape_RMSE', ''), ('SI_mean', ''),
]

# Scalar family: a lone 'hs' or 'shape' checkpoint has no physical spectrum to
# derive Hs_RMSE/Tm02_RMSE/Shape_RMSE/SI from — these are nn/evaluate.py's own
# metrics for that target, evaluated directly (see load_scalar).
#
# 'shape' deliberately shows Shape_RMSE only, not the usual RMSE/R2/CC:
# nn/evaluate.py computes those top-level metrics in LOG-space for the
# 'shape' target (see its docstring), so they are NOT comparable across an
# arbitrary set of rows — in particular not against a linear_baseline:shape
# row (utils/linear_baseline.py never leaves physical space). Shape_RMSE is
# explicitly exp()'d back to physical units in nn/evaluate.py and is also
# what utils/linear_baseline.py reports under that same key, so it's safe
# for every row regardless of source.
SUMMARY_METRICS_SCALAR = {
    'hs': [('RMSE', 'm'), ('Hs_MAPE', '%'), ('R2', ''), ('CC', '')],
    'shape': [('Shape_RMSE', '')],
}

# Bottom-line Skill Score row per scalar target — 'hs's overall_SS is
# physical (safe to show); 'shape's overall_SS is log-space (see above), so
# Shape_SS (physical, same reasoning as Shape_RMSE) is shown instead.
SS_KEY_SCALAR = {'hs': 'overall_SS', 'shape': 'Shape_SS'}


def load_linear_baseline_checkpoint(project_root, target, lead):
    """Load the checkpoint scripts/train_linear_baseline.py saves —
    {'coeffs', 'order', 'ridge', 'target', 'lead_time_steps', 'freqs',
    'buoy_id'} — for a given target/lead. Unlike find_checkpoint (transformer
    checkpoints), this isn't seed- or experiment-namespaced: the linear
    baseline only depends on (buoy, target, lead), not on channel_set/
    aux_set/architecture, so results/linear_baseline/ is shared across every
    --experiment comparison for that target/lead.
    """
    path = (project_root / 'results' / 'linear_baseline' / target
             / f'lead_{lead}h' / 'linear_baseline_final.pt')
    if not path.exists():
        raise FileNotFoundError(
            f"No linear baseline checkpoint at {path} — run "
            f"scripts/train_linear_baseline.py first (target={target!r}, lead={lead}h).")
    print(f"Loaded linear baseline checkpoint from {path}")
    return torch.load(path, map_location='cpu', weights_only=False)


def load_scalar(project_root, spec, lead, seed, density, alpha_1, alpha_2, r_1, r_2, wind,
                freqs, device, channel_set, aux_set, return_arrays=False):
    """Full test-set nn/evaluate.py metrics for a lone 'hs' or 'shape' checkpoint —
    no spectrum recombination, just that model's own predictions vs its own target.

    spec['name'] == 'linear_baseline' is special-cased: loads and scores the
    utils/linear_baseline.py checkpoint instead of a transformer one — see
    load_linear_baseline_checkpoint and the module docstring.

    return_arrays : bool, mirrors nn/evaluate.py::evaluate()'s own flag —
        if True, additionally returns (pred, true, pers, t0_start): physical
        arrays, shape (N, lead_time, 1|num_freqs), plus the test-split-
        relative row index of sample 0's forecast start (same convention as
        nn/spectrum_eval.py::eval_single_density's t0_start). For
        target=='shape' the transformer's raw output is log-shape (see
        nn/evaluate.py's docstring) and is exp()'d here back to the physical
        unit-area shape so it lines up with linear_baseline's always-
        physical arrays; target=='hs' is already physical either way, but
        has no frequency axis so isn't meaningful to plot as a curve — only
        the 'shape' target actually uses this (see main()'s example-shape-
        curve plot).
    """
    if spec['name'] == 'linear_baseline':
        ckpt = load_linear_baseline_checkpoint(project_root, spec['target'], lead)
        if return_arrays:
            metrics, (pred, true, pers) = evaluate_coeffs(
                density, freqs, ckpt['coeffs'], ckpt['order'], ckpt['lead_time_steps'],
                spec['target'], return_arrays=True)
            return metrics, ckpt['lead_time_steps'], pred, true, pers, ckpt['order']
        metrics = evaluate_coeffs(density, freqs, ckpt['coeffs'], ckpt['order'],
                                  ckpt['lead_time_steps'], spec['target'])
        return metrics, ckpt['lead_time_steps']

    ckpt, _ = find_checkpoint(project_root, spec['name'], spec['target'], 1, lead, seed)
    model = build_model(ckpt, freqs, device, channel_set, aux_set)
    freq_means = ckpt['freq_means'].to(device)
    shape_means = ckpt['shape_means'].to(device) if ckpt.get('shape_means') is not None else None
    lead_time_steps = ckpt['lead_time_steps']
    params = ckpt['params']
    _, _, test_loader, _, _, _, _, _ = _prepare_dataloaders(
        density, alpha_1, alpha_2, r_1, r_2, params['seq_len'], lead_time_steps,
        params['batch_size'], spec['target'], shuffle_seed=seed, wind=wind,
        channel_set=channel_set, aux_set=aux_set)
    if return_arrays:
        metrics, (pred, true, pers) = evaluate(model, test_loader, device, freqs,
                           lead_time=lead_time_steps, freq_means=freq_means,
                           shape_means=shape_means, return_arrays=True)
        if spec['target'] == 'shape':
            pred, true, pers = torch.exp(pred), torch.exp(true), torch.exp(pers)
        return metrics, lead_time_steps, pred.numpy(), true.numpy(), pers.numpy(), params['seq_len']
    metrics = evaluate(model, test_loader, device, freqs,
                       lead_time=lead_time_steps, freq_means=freq_means,
                       shape_means=shape_means)
    return metrics, lead_time_steps


def print_scalar_comparison(results, target):
    labels = [r['label'] for r in results]
    col_w = 16
    metrics = SUMMARY_METRICS_SCALAR[target]
    print(f"\n{'Metric':<{col_w}}" + ''.join(f"{l:>18}" for l in labels))
    print('-' * (col_w + 18 * len(labels)))
    for key, unit in metrics:
        row = f"{key:<{col_w}}"
        for r in results:
            row += f"{r['metrics'][key]:>15.4f} {unit:<2}"
        print(row)
    ss_key = SS_KEY_SCALAR[target]
    print(f"{ss_key:<{col_w}}" + ''.join(f"{r['metrics'][ss_key]:>18.4f}" for r in results))

    if len(results) > 1:
        base = results[0]
        others = results[1:]
        print(f"\nΔ vs {base['label']} (per-metric, later experiment minus baseline):")
        print(f"{'Metric':<{col_w}}" + ''.join(f"{r['label']:>18}" for r in others))
        for key, unit in metrics:
            row = f"{key:<{col_w}}"
            for r in others:
                row += f"{(r['metrics'][key] - base['metrics'][key]):>+15.4f} {unit:<2}"
            print(row)


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


def plot_example_shape_curves(results, t0_examples, freqs_np, lead_time_steps, out_path):
    """Predicted vs true (vs persistence) unit-area shape curves E(f)/m0 at
    the FINAL forecast step, for a handful of example forecast-start times —
    the scalar 'shape'-family counterpart to plot_example_spectra (which
    needs a physical density spectrum the scalar family doesn't have; 'hs'
    has no frequency axis at all, so this only applies to 'shape'). Kept as
    its own function rather than parameterizing plot_example_spectra — the
    two differ enough in axis label/units/title that sharing one function
    would need almost as many parameters as just duplicating the ~20 lines,
    and this way the working spectrum-family plot is untouched.

    Expects each result dict to carry 'true_common'/'pers_common'/
    'pred_common' arrays already sliced down to t0_examples (see main()).
    """
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
        ax.set_ylabel('Shape E(f)/m₀ (unit-area)')
        ax.set_title(f't0={t0} (+{lead_time_steps}h)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    labels = [r['label'] for r in results]
    fig.suptitle(' vs '.join(labels) + f' — example unit-area shape curves at the {lead_time_steps}h horizon')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    device = get_device()
    print(f"Device: {device}")

    # linear_baseline gets its own fixed color (not a categorical slot) so
    # it reads visually as "the baseline", not just another experiment, and
    # so its color stays stable regardless of where it falls in the --experiment
    # order or how many real experiments are also being compared.
    real_experiments = [s for s in args.experiments if s['name'] != 'linear_baseline']
    if len(real_experiments) > len(CATEGORICAL_PALETTE):
        print(f"  warning: {len(real_experiments)} experiments requested but only "
              f"{len(CATEGORICAL_PALETTE)} categorical colors are defined — colors will repeat.")
    for i, spec in enumerate(real_experiments):
        spec['color'] = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
    for spec in args.experiments:
        if spec['name'] == 'linear_baseline':
            spec['color'] = COLOR_LINEAR

    # Must match the buoy every checkpoint-producing script (scripts/train.py,
    # scripts/optimize.py, scripts/infer.py, scripts/train_linear_baseline.py)
    # actually trains on — a prior commit had silently drifted this to '42056'
    # with no comment/rationale, which meant every comparison here was scored
    # against the wrong buoy's data.
    density, alpha_1, alpha_2, r_1, r_2, wind = pd.read_pickle(project_root / 'buoy_data' / '32012' / 'processed_data.pkl')
    freqs = get_freqs(density)
    freqs_np = freqs.cpu().numpy() if torch.is_tensor(freqs) else np.asarray(freqs)

    out_dir = Path(args.out_dir) if args.out_dir else (
        project_root / 'results' / 'comparisons' / '_vs_'.join(s['name'] for s in args.experiments) / f'lead_{args.lead}h')

    targets = {spec['target'] for spec in args.experiments}
    if targets <= SCALAR_TARGETS:
        target = next(iter(targets))
        # 'shape' has a frequency axis (unit-area E(f)/m0), so example-curve
        # plotting is meaningful there; 'hs' is a bare scalar with nothing to
        # plot as a curve, so it stays table-only (see the else-branch message
        # below).
        want_arrays = (target == 'shape')
        results = []
        for spec in args.experiments:
            print(f"\nEvaluating {spec['label']} (target={spec['target']})")
            loaded = load_scalar(
                project_root, spec, args.lead, args.seed, density, alpha_1, alpha_2, r_1, r_2,
                wind, freqs, device, args.channel_set, args.aux_set, return_arrays=want_arrays)
            if want_arrays:
                metrics, lead_time_steps, pred, true, pers, t0_start = loaded
                results.append({**spec, 'metrics': metrics, 'lead_time_steps': lead_time_steps,
                                'pred': pred, 'true': true, 'pers': pers, 't0_start': t0_start})
            else:
                metrics, lead_time_steps = loaded
                results.append({**spec, 'metrics': metrics, 'lead_time_steps': lead_time_steps})

        lead_time_steps = results[0]['lead_time_steps']
        for r in results[1:]:
            if r['lead_time_steps'] != lead_time_steps:
                raise ValueError(f"lead_time_steps mismatch: {results[0]['label']}={lead_time_steps} "
                                 f"{r['label']}={r['lead_time_steps']} — comparison requires the same forecast horizon.")

        print_scalar_comparison(results, target)

        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump({
                'experiments': [
                    {'name': r['name'], 'label': r['label'], 'target': r['target'], 'metrics': r['metrics']}
                    for r in results
                ]
            }, f, indent=2)
        print(f"\nSaved metrics.json to {out_dir}")

        if want_arrays:
            print("\nGenerating plots...")
            # Same "common t0 range" logic as the spectrum family's
            # example_spectra.png (see main()'s spectrum branch below) —
            # only t0 values covered by every experiment's encoder window
            # have directly comparable True/Persistence references.
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
                plot_example_shape_curves(results, t0_examples, freqs_np, lead_time_steps,
                                          out_dir / 'example_shapes.png')
            else:
                print("  no forecast-start range common to every experiment — skipping example_shapes.png")
        else:
            print(f"target={target!r} has no frequency axis — skipping example_shapes.png "
                  f"(and summary_bars/per_step/si_per_bin, spectrum family only).")
        print(f"\nDone. All outputs under {out_dir}")
        return

    def load(spec):
        if spec['name'] == 'linear_baseline':
            # target == 'density' is enforced by parse_experiment_spec (no
            # 'combined' variant). Always physical, like every other source
            # eval_single_density/eval_combined can return — see
            # nn/spectrum_eval.py::compute_density_metrics's docstring.
            ckpt = load_linear_baseline_checkpoint(project_root, 'density', args.lead)
            pred, true, pers = forecast_coeffs(
                density, freqs, ckpt['coeffs'], ckpt['order'], ckpt['lead_time_steps'], 'density')
            return pred, true, pers, ckpt['lead_time_steps'], ckpt['order']
        if spec['target'] == 'density':
            # deltat is not yet a CLI option here — 1 matches this script's
            # previous (undownsampled) behavior; find_checkpoint still falls
            # back to the pre-deltat results path for older experiments.
            ckpt, _ = find_checkpoint(project_root, spec['name'], 'density', 1, args.lead, args.seed)
            return eval_single_density(ckpt, density, alpha_1, alpha_2, r_1, r_2, wind,
                                       freqs, device, args.channel_set, args.aux_set, args.seed)
        else:
            return eval_combined(project_root, spec['name'], 1, args.lead, args.seed,
                                 density, alpha_1, alpha_2, r_1, r_2, wind,
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
