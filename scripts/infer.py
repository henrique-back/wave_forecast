"""
Inspect a trained model's predictions on the held-out test set.

Loads a checkpoint saved by scripts/train.py (final_model_seed{N}.pt), runs
autoregressive inference on selected test-set samples, prints per-sample bulk
parameters (Hs, Tm02) across the forecast horizon, and plots predicted vs
true vs persistence.

Usage:
    python scripts/infer.py --experiment weightedmeanSS_conv_freqemb_v3 --lead 6
    python scripts/infer.py --experiment weightedmeanSS_conv_freqemb_v3 --lead 6 \
        --index 42 --save
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from utils import get_freqs, compute_bulk_params, get_start_token
from nn import WaveHeightBaselineNN
from nn.optimization import _prepare_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--experiment', required=True,
                   help="EXPERIMENT_NAME the checkpoint was trained under (results/{experiment}/...)")
    p.add_argument('--target', default='density', choices=['density', 'hs'])
    p.add_argument('--deltat', type=int, default=1)
    p.add_argument('--lead', type=int, required=True, help='Lead time in hours')
    p.add_argument('--seed', type=int, default=42, help='Seed suffix of the checkpoint to load')
    p.add_argument('--index', type=int, default=None,
                   help='Specific test-set sample index to inspect (overrides --n-samples/--random)')
    p.add_argument('--n-samples', type=int, default=3,
                   help='Number of samples to inspect when --index is not given')
    p.add_argument('--random', action='store_true',
                   help='Pick random sample indices instead of evenly spaced across the test set')
    p.add_argument('--n-steps', type=int, default=4,
                   help='Number of forecast-horizon spectrum snapshots to plot per sample (density target only)')
    p.add_argument('--save', action='store_true',
                   help='Also save each figure as PNG under the checkpoint folder')
    return p.parse_args()


def load_checkpoint(experiment, target, deltat, lead, seed, project_root):
    results_folder = (project_root / 'results' / experiment / target
                      / f'deltat_{deltat}' / f'lead_{lead}h')
    ckpt_path = results_folder / f'final_model_seed{seed}.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} — run scripts/train.py for this "
            f"experiment/lead/seed first.")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt, results_folder


def build_model(ckpt, freqs, device):
    params = ckpt['params']
    embed_dim = params['head_dim'] * params['nhead']
    model = WaveHeightBaselineNN(
        num_freqs=len(freqs),
        freqs=freqs,
        target=ckpt['target'],
        dropout=params['dropout'],
        nhead=params['nhead'],
        num_encoder_layers=params['num_encoder_layers'],
        num_decoder_layers=params['num_decoder_layers'],
        embed_dim=embed_dim,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def select_indices(n_total, n_samples, index, random_pick, seed):
    if index is not None:
        return [index]
    n_samples = min(n_samples, n_total)
    if random_pick:
        rng = np.random.default_rng(seed)
        return sorted(rng.choice(n_total, size=n_samples, replace=False).tolist())
    return sorted(set(np.linspace(0, n_total - 1, n_samples, dtype=int).tolist()))


def plot_density_sample(idx, freqs_np, steps, pred_phys, true_phys, pers_phys,
                        hs_pred, hs_true, hs_pers):
    lead_time = pred_phys.shape[0]
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(4 * (len(steps) + 1), 4))

    ax = axes[0]
    x = np.arange(1, lead_time + 1)
    ax.plot(x, hs_true, 'k-o', label='True', markersize=4)
    ax.plot(x, hs_pred, 'C0-o', label='Predicted', markersize=4)
    ax.plot(x, hs_pers, 'C1--', label='Persistence', markersize=4)
    ax.set_xlabel('Forecast step')
    ax.set_ylabel('Hs (m)')
    ax.set_title(f'Sample {idx} — Hs over horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax, step in zip(axes[1:], steps):
        ax.plot(freqs_np, true_phys[step], 'k-', label='True')
        ax.plot(freqs_np, pred_phys[step], 'C0-', label='Predicted')
        ax.plot(freqs_np, pers_phys[step], 'C1--', label='Persistence')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('E(f) (m²/Hz)')
        ax.set_title(f'Step {step + 1} (Hs true={hs_true[step]:.2f} pred={hs_pred[step]:.2f})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_hs_sample(idx, hs_pred, hs_true, hs_pers):
    lead_time = len(hs_true)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, lead_time + 1)
    ax.plot(x, hs_true, 'k-o', label='True', markersize=4)
    ax.plot(x, hs_pred, 'C0-o', label='Predicted', markersize=4)
    ax.plot(x, hs_pers, 'C1--', label='Persistence', markersize=4)
    ax.set_xlabel('Forecast step')
    ax.set_ylabel('Hs (m)')
    ax.set_title(f'Sample {idx} — Hs over horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / 'buoy_data' / 'processed_data.pkl'
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found — run scripts/data_processing.py first.")
    density, alpha_1, alpha_2, r_1 = pd.read_pickle(file_path)
    freqs = get_freqs(density)
    freqs_np = freqs.numpy()

    ckpt, results_folder = load_checkpoint(args.experiment, args.target, args.deltat,
                                            args.lead, args.seed, project_root)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(ckpt, freqs, device)
    freq_means = ckpt['freq_means'].to(device)
    lead_time_steps = ckpt['lead_time_steps']
    params = ckpt['params']

    density_d = density[::args.deltat]
    alpha_1_d = alpha_1[::args.deltat]
    alpha_2_d = alpha_2[::args.deltat]
    r_1_d = r_1[::args.deltat]

    # shuffle_seed only affects the (unused) train_loader's shuffle order —
    # the test split itself is deterministic and never shuffled.
    _, _, test_loader, _, _ = _prepare_dataloaders(
        density_d, alpha_1_d, alpha_2_d, r_1_d,
        params['seq_len'], lead_time_steps, params['batch_size'],
        args.target, shuffle_seed=args.seed)
    test_dataset = test_loader.dataset

    n_total = len(test_dataset)
    indices = select_indices(n_total, args.n_samples, args.index, args.random, args.seed)
    print(f"Inspecting {len(indices)} test sample(s) out of {n_total}: {indices}")

    save_dir = None
    if args.save:
        save_dir = results_folder / 'inference_plots'
        save_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for idx in indices:
            X, y_true = test_dataset[idx]
            X = X.unsqueeze(0).to(device)
            y_true = y_true.unsqueeze(0).to(device)

            y_pred = model.infer(X, freqs, lead_time_steps, freq_means=freq_means)
            start_token = get_start_token(X, args.target, freqs, device, freq_means=freq_means)
            persistence = start_token.unsqueeze(1).expand(-1, y_true.shape[1], -1)

            if args.target == 'density':
                fm = freq_means.cpu().numpy()
                pred_phys = y_pred.cpu().numpy()[0] * fm
                true_phys = y_true.cpu().numpy()[0] * fm
                pers_phys = persistence.cpu().numpy()[0] * fm

                hs_pred, tm02_pred = compute_bulk_params(pred_phys[np.newaxis], freqs_np)
                hs_true, tm02_true = compute_bulk_params(true_phys[np.newaxis], freqs_np)
                hs_pers, tm02_pers = compute_bulk_params(pers_phys[np.newaxis], freqs_np)
                hs_pred, hs_true, hs_pers = hs_pred[0], hs_true[0], hs_pers[0]
                tm02_pred, tm02_true = tm02_pred[0], tm02_true[0]

                print(f"\nSample {idx}:")
                print(f"  {'step':>4} {'Hs true':>8} {'Hs pred':>8} {'Hs pers':>8} "
                      f"{'Tm02 true':>10} {'Tm02 pred':>10}")
                for s in range(lead_time_steps):
                    print(f"  {s+1:>4} {hs_true[s]:>8.3f} {hs_pred[s]:>8.3f} {hs_pers[s]:>8.3f} "
                          f"{tm02_true[s]:>10.3f} {tm02_pred[s]:>10.3f}")

                n_snap = min(args.n_steps, lead_time_steps)
                steps = sorted(set(np.linspace(0, lead_time_steps - 1, n_snap, dtype=int).tolist()))
                fig = plot_density_sample(idx, freqs_np, steps, pred_phys, true_phys, pers_phys,
                                          hs_pred, hs_true, hs_pers)
            else:
                hs_pred = y_pred.cpu().numpy()[0, :, 0]
                hs_true = y_true.cpu().numpy()[0, :, 0]
                hs_pers = persistence.cpu().numpy()[0, :, 0]
                print(f"\nSample {idx}:")
                print(f"  {'step':>4} {'Hs true':>8} {'Hs pred':>8} {'Hs pers':>8}")
                for s in range(lead_time_steps):
                    print(f"  {s+1:>4} {hs_true[s]:>8.3f} {hs_pred[s]:>8.3f} {hs_pers[s]:>8.3f}")
                fig = plot_hs_sample(idx, hs_pred, hs_true, hs_pers)

            if save_dir is not None:
                out_path = save_dir / f'sample_{idx}.png'
                fig.savefig(out_path, dpi=150)
                print(f"  Saved plot → {out_path}")

    plt.show()


if __name__ == '__main__':
    main()
