"""Evaluation utilities for the wave forecast transformer model."""
import torch
from tqdm import tqdm
from torchmetrics.functional import (
    mean_absolute_percentage_error,
    pearson_corrcoef
)
from utils import RMSELoss, get_start_token


def evaluate(model, dataloader, device='cpu', freqs=None, lead_time=None):
    """
    Evaluate model using autoregressive inference.

    Persistence baseline: predict that every future step equals the last
    observed value at the end of the encoder window (i.e. the start token).
    This is constructed without any leakage — only information available at
    forecast time is used.

    Parameters:
    - model     : PyTorch model with an infer() method
    - dataloader: DataLoader yielding (X_batch, y_batch)
    - device    : 'cpu' or 'cuda'
    - freqs     : frequency tensor required for Hs computation and infer()
    - lead_time : number of steps to forecast (passed to model.infer)

    Returns dict with keys:
        'RMSE'               : overall RMSE (all steps, all samples flattened)
        'MAPE'               : overall MAPE (flattened)
        'CC'                 : Pearson correlation (flattened)
        'Bias'               : mean error = mean(pred - true), flattened
        'R2'                 : coefficient of determination, flattened
        'per_step_RMSE'      : list[float], one RMSE per forecast step
        'per_step_SS'        : list[float], Skill Score vs persistence per step
                               SS = 1 - RMSE_model / RMSE_persistence
                               SS > 0: model beats persistence
                               SS = 1: perfect forecast
                               SS < 0: persistence is better
        'per_step_Bias'      : list[float], mean error per forecast step
        'per_step_R2'        : list[float], R² per forecast step
        'overall_SS'         : float, Skill Score computed on overall flattened RMSE

    Note on optimization equivalence
    ---------------------------------
    Skill Score is a monotone transformation of RMSE given a fixed persistence
    baseline.  However, the persistence RMSE is NOT constant across Optuna
    trials because `seq_len` is a hyperparameter — different seq_len values
    produce different sample windows and therefore different last-observed
    values.  Minimizing RMSE is therefore NOT strictly equivalent to maximizing
    Skill Score across trials.  The optimizer should consequently use Skill
    Score (or equivalently mean per-step RMSE relative to persistence) as its
    objective for a fair comparison.  Currently, mean per-step RMSE is used,
    which is a reasonable proxy but may favour configurations with easier
    persistence baselines.  Flag this if you want to switch the Optuna
    objective to mean per-step Skill Score.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_persistence = []
    rmse_fn = RMSELoss()

    with torch.no_grad():
        for src, y_batch in tqdm(dataloader):
            src = src.to(device)
            y_batch = y_batch.to(device)

            # Persistence forecast: last observed value broadcast over all steps
            # get_start_token returns shape (batch, 1) for hs or (batch, num_freqs) for density
            start_token = get_start_token(src, model.target, freqs, device)
            # shape → (batch, 1, output_dim) → (batch, lead_time, output_dim)
            persistence = start_token.unsqueeze(1).expand(-1, y_batch.shape[1], -1)

            # Autoregressive model inference — no ground truth in decoder
            y_pred = model.infer(src, freqs, lead_time)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())
            all_persistence.append(persistence.cpu())

    # Shape: (total_samples, lead_time, output_dim)
    y_pred_all  = torch.cat(all_preds, dim=0)
    y_true_all  = torch.cat(all_targets, dim=0)
    y_pers_all  = torch.cat(all_persistence, dim=0)

    # Per-step metrics
    per_step_rmse = []
    per_step_rmse_pers = []
    per_step_ss = []
    per_step_bias = []
    per_step_r2   = []

    for step in range(y_pred_all.shape[1]):
        pred_s = y_pred_all[:, step, :].flatten()
        true_s = y_true_all[:, step, :].flatten()
        pers_s = y_pers_all[:, step, :].flatten()

        rmse_s = rmse_fn(pred_s, true_s).item()
        rmse_p = rmse_fn(pers_s, true_s).item()
        ss_s   = 1.0 - rmse_s / rmse_p if rmse_p > 0 else float('nan')
        bias_s = (pred_s - true_s).mean().item()
        ss_res = ((true_s - pred_s) ** 2).sum()
        ss_tot = ((true_s - true_s.mean()) ** 2).sum()
        r2_s   = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')

        per_step_rmse.append(rmse_s)
        per_step_rmse_pers.append(rmse_p)
        per_step_ss.append(ss_s)
        per_step_bias.append(bias_s)
        per_step_r2.append(r2_s)

    # Overall (flattened) metrics
    y_pred_flat = y_pred_all.flatten()
    y_true_flat = y_true_all.flatten()
    y_pers_flat = y_pers_all.flatten()

    rmse       = rmse_fn(y_pred_flat, y_true_flat).item()
    rmse_pers  = rmse_fn(y_pers_flat, y_true_flat).item()
    mape       = mean_absolute_percentage_error(y_pred_flat, y_true_flat).item()
    cc         = pearson_corrcoef(y_pred_flat, y_true_flat).item()
    overall_ss = 1.0 - rmse / rmse_pers if rmse_pers > 0 else float('nan')
    bias       = (y_pred_flat - y_true_flat).mean().item()
    ss_res     = ((y_true_flat - y_pred_flat) ** 2).sum()
    ss_tot     = ((y_true_flat - y_true_flat.mean()) ** 2).sum()
    r2         = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')

    return {
        'RMSE'               : rmse,
        'MAPE'               : mape,
        'CC'                 : cc,
        'Bias'               : bias,
        'R2'                 : r2,
        'per_step_RMSE'      : per_step_rmse,
        'per_step_RMSE_pers' : per_step_rmse_pers,
        'per_step_SS'        : per_step_ss,
        'per_step_Bias'      : per_step_bias,
        'per_step_R2'        : per_step_r2,
        'overall_SS'         : overall_ss,
    }
