import math
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from nn import WaveSpectralDataset, WaveHeightBaselineNN, prepare_X, prepare_y, train_one_epoch, evaluate
from utils import set_seed


def _seed_worker(worker_id):
    torch.manual_seed(42 + worker_id)


def _normalize(train_df, *other_dfs, mode='zscore'):
    """Fit normalization on train_df and apply to all DataFrames.

    mode='zscore': subtract mean, divide by std.  May produce negative values —
        use for channels that are not fed into physical computations (alpha, r1).
    mode='scale':  divide by per-column mean only.  Preserves non-negativity —
        required for spectral density, which is passed to compute_hs / sqrt().
    """
    if mode == 'zscore':
        mean = train_df.mean()
        std = train_df.std().clip(lower=1e-8)
        return tuple((df - mean) / std for df in (train_df, *other_dfs))
    else:  # 'scale'
        mean = train_df.mean().clip(lower=1e-8)
        return tuple(df / mean for df in (train_df, *other_dfs))


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
        'weighted_mean_SS' : exponentially-weighted mean per-step Skill Score
                             (recommended default — robust to variable seq_len)
        'overall_SS'       : Skill Score on flattened all-step RMSE
        'RMSE'             : negative overall RMSE
        'Hs_RMSE'          : negative Hs RMSE (density target only)
        'Tm02_RMSE'        : negative Tm02 RMSE (density target only)
        'Shape_RMSE'       : negative spectral shape RMSE (density target only)
        'SI_mean'          : negative mean Scatter Index (density target only)
    """
    if objective_metric == 'weighted_mean_SS':
        return _weighted_mean_ss(metrics['per_step_SS'])
    elif objective_metric == 'overall_SS':
        return metrics['overall_SS']
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
    else:
        raise ValueError(
            f"Unknown objective_metric {objective_metric!r}. Valid: "
            "'weighted_mean_SS', 'overall_SS', 'RMSE', 'Hs_RMSE', "
            "'Tm02_RMSE', 'Shape_RMSE', 'SI_mean'"
        )


def _train_model(model, train_loader, val_loader, device, freqs, freq_means,
                  target, lead_time, lr, weight_decay, objective_metric,
                  num_epochs=100, patience=10, trial=None):
    """Run the scheduled-sampling training loop with early stopping.

    Shared by objective() (Optuna trial) and scripts/train.py (fixed-config
    final retrain) so the two never drift apart. When `trial` is given,
    reports the per-epoch score to Optuna and prunes on its signal; this is
    the only behavioural difference between the two callers.

    Returns (best_val_score, best_val_metrics, best_model_state).
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # patience=3 so the LR is halved 7 epochs before early stopping fires (at
    # patience=10), giving the model meaningful time to benefit from the new LR.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    # tf_ratio decays from 1.0 to 0.0 over 4×patience epochs.  With early
    # stopping at patience=10, a run going to epoch ~20 will have tf_ratio
    # ≈ 0.5 — half its training steps use the model's own predictions, which
    # meaningfully closes the teacher-forcing / autoregressive distribution gap.
    tf_decay_epochs = 4 * patience

    best_val_score = float('-inf')
    best_val_metrics = None
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        tf_ratio = max(0.0, 1.0 - epoch / tf_decay_epochs)

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, freqs,
                                        tf_ratio=tf_ratio, freq_means=freq_means)
        val_metrics   = evaluate(model, val_loader, device, freqs,
                                  lead_time=lead_time, freq_means=freq_means)

        val_score = _compute_val_score(val_metrics, objective_metric)

        scheduler.step(val_score)

        bulk_str = ""
        if target == 'density' and 'Hs_RMSE' in val_metrics:
            bulk_str = (f" | Val Hs_RMSE: {val_metrics['Hs_RMSE']:.4f}"
                        f" | Val Hs_Bias: {val_metrics['Hs_Bias']:+.4f}"
                        f" | Val Tm02_RMSE: {val_metrics['Tm02_RMSE']:.4f}"
                        f" | Val Tm02_Bias: {val_metrics['Tm02_Bias']:+.4f}"
                        f" | Val Shape_RMSE: {val_metrics['Shape_RMSE']:.4f}"
                        f" (masked: {val_metrics['Shape_masked_samples']})"
                        f" | Val SI_mean: {val_metrics['SI_mean']:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train RMSE: {train_metrics['RMSE']:.4f} | "
              f"Val RMSE: {val_metrics['RMSE']:.4f} | "
              f"Val Hs_MAPE: {val_metrics['Hs_MAPE']:.2f}% | "
              f"Val CC: {val_metrics['CC']:.4f} | "
              f"Val {objective_metric}: {val_score:.4f} | "
              f"tf_ratio: {tf_ratio:.2f}"
              + bulk_str)

        if trial is not None:
            trial.report(val_score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if val_score > best_val_score:
            best_val_score = val_score
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


def _prepare_dataloaders(density, alpha_1, alpha_2, r_1, seq_len, lead_time, batch_size,
                          target, shuffle_seed):
    """Split, normalise, and window the four spectral channels into DataLoaders.

    Shared by objective() and scripts/train.py so the train/val/test split and
    normalisation are always computed identically for a given (seq_len,
    lead_time, target) — a final retrain must see exactly the same splits the
    hyperparameter search saw for its metrics to be comparable.

    Returns (train_loader, val_loader, test_loader, freq_means, num_freqs).
    freq_means is the per-frequency training-split mean μ(f) — the
    denormalisation key E_phys = Ẽ * μ(f) used throughout training/eval.
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

    # Compute per-frequency training mean μ(f) BEFORE normalising.
    # This tensor is the denormalisation key: E_phys = Ẽ * μ(f).
    # It is passed to the training loop and evaluator so that all spectral
    # integrations (Hs, Tm02, shape error, SI) and the density training loss
    # operate on physical m² Hz⁻¹ values, not on normalised dimensionless ones.
    freq_means = torch.tensor(
        train_density.mean().clip(lower=1e-8).values, dtype=torch.float32
    )  # shape: (num_freqs,)

    # For the Hs target, build sequence targets from PHYSICAL (pre-normalisation)
    # density so that y_batch and the persistence start token are both in metres.
    # For the density target, targets are the normalised spectra (model operates
    # in normalised space; freq_means is applied externally at loss/metric time).
    if target == 'hs':
        train_y = prepare_y(train_density, seq_len, lead_time, target='hs')
        val_y   = prepare_y(val_density,   seq_len, lead_time, target='hs')
        test_y  = prepare_y(test_density,  seq_len, lead_time, target='hs')

    # Normalize inputs — fit on training data, apply to all splits.
    # Density uses scale-only normalization (divide by per-frequency training mean)
    # to preserve non-negativity: compute_hs calls sqrt(trapz(density)) and would
    # produce NaN if density went negative from z-scoring.
    # Alpha and r1 have no downstream physical constraint so z-score is safe.
    train_density, val_density, test_density = _normalize(
        train_density, val_density, test_density, mode='scale')
    train_alpha1, val_alpha1, test_alpha_1 = _normalize(train_alpha1, val_alpha1, test_alpha_1)
    train_alpha2, val_alpha2, test_alpha_2 = _normalize(train_alpha2, val_alpha2, test_alpha_2)
    train_r1, val_r1, test_r1             = _normalize(train_r1, val_r1, test_r1)

    # Encoder input sequences always use normalised density.
    train_X = prepare_X(train_density, train_alpha1, train_alpha2, train_r1, seq_len, lead_time)
    val_X   = prepare_X(val_density,   val_alpha1,   val_alpha2,   val_r1,   seq_len, lead_time)
    test_X  = prepare_X(test_density,  test_alpha_1, test_alpha_2, test_r1,  seq_len, lead_time)

    if target == 'density':
        train_y = prepare_y(train_density, seq_len, lead_time, target='density')
        val_y   = prepare_y(val_density,   seq_len, lead_time, target='density')
        test_y  = prepare_y(test_density,  seq_len, lead_time, target='density')

    # DataLoaders — generator seeded explicitly so shuffle order is reproducible
    # for a given shuffle_seed (trial.number during HPO; a chosen seed during
    # final retrain).
    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    train_loader = DataLoader(WaveSpectralDataset(train_X, train_y), batch_size=batch_size, shuffle=True,
                              worker_init_fn=_seed_worker, generator=g)
    val_loader   = DataLoader(WaveSpectralDataset(val_X, val_y), batch_size=batch_size, shuffle=False,
                              worker_init_fn=_seed_worker, generator=g)
    test_loader  = DataLoader(WaveSpectralDataset(test_X, test_y), batch_size=batch_size, shuffle=False,
                              worker_init_fn=_seed_worker, generator=g)

    return train_loader, val_loader, test_loader, freq_means, train_X.shape[2]


def objective(trial, *, density, alpha_1, alpha_2, r_1, freqs, lead_time, target,
              objective_metric='weighted_mean_SS', results_folder=None):
    # set_seed() is called once at script level — do NOT call it here.
    # Resetting the RNG inside objective() makes every trial start from the same
    # random state, collapsing the variance Optuna needs to learn from.

    # Sample hyperparameters
    seq_len = trial.suggest_categorical('seq_len', [12, 24, 48, 96])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    # embed_dim derived as head_dim × nhead so it is always divisible by nhead.
    # nhead starts at 4 so the minimum embed_dim is 8×4=32.
    head_dim = trial.suggest_categorical('head_dim', [8, 16, 32])
    nhead = trial.suggest_categorical('nhead', [4, 8])
    embed_dim = head_dim * nhead
    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4)
    num_decoder_layers = trial.suggest_int('num_decoder_layers', 1, 4)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)

    # Safety net: embed_dim must be divisible by nhead (guaranteed by construction
    # above, but kept to catch any future reparameterization changes).
    if embed_dim % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    # --- Data preparation ---
    # DataLoader shuffle seeded per trial (using trial.number) so shuffle order
    # differs between trials while remaining reproducible within each trial.
    train_loader, val_loader, test_loader, freq_means, num_freqs = _prepare_dataloaders(
        density, alpha_1, alpha_2, r_1, seq_len, lead_time, batch_size, target,
        shuffle_seed=trial.number)

    # --- Model ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on device: {device}')

    model = WaveHeightBaselineNN(
        num_freqs=num_freqs,
        freqs=freqs,
        target=target,
        dropout=dropout,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        embed_dim=embed_dim,
    )
    model = model.to(device)

    best_val_score, best_val_metrics, best_model_state = _train_model(
        model, train_loader, val_loader, device, freqs, freq_means,
        target, lead_time, lr, weight_decay, objective_metric,
        num_epochs=100, patience=10, trial=trial)

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
                        'Shape_RMSE', 'Shape_masked_samples',
                        'SI_per_bin', 'SI_mean']:
                if key in best_val_metrics:
                    trial.set_user_attr(f'val_{key}', best_val_metrics[key])

    # Restore best-epoch weights before test evaluation so the reported test
    # metrics correspond to the same model that produced best_val_score.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device, freqs,
                             lead_time=lead_time, freq_means=freq_means)
    print(f"Final test metrics: {test_metrics}")
    return best_val_score
