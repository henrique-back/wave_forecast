import torch
from tqdm import tqdm
from utils import compute_hs, RMSELoss
from torchmetrics.functional import (
    mean_absolute_percentage_error,
    pearson_corrcoef
)

def evaluate(model, dataloader, device='cpu', freqs=None):
    """
    Evaluate model with RMSE, MAPE, Pearson CC per horizon and overall.

    Returns:
    - metrics_per_horizon: dict with keys 'RMSE', 'MAPE', 'CC', each shape (lead_time,)
    - metrics_overall: dict with single float values averaged across horizons
    """
    model.eval()
    all_preds = []
    all_targets = []
    rmse_fn = RMSELoss()

    with torch.no_grad():
        for src, y_batch in tqdm(dataloader):
            src = src.to(device)
            y_batch = y_batch.to(device)

            # Prepare decoder input
            last_spectrum = src[:, -1, :, 0]  # spectrum channel
            if model.target == 'hs':
                hs = compute_hs(last_spectrum.cpu().numpy(), freqs.cpu().numpy())
                hs = torch.from_numpy(hs).to(device).float()
                start_token = hs.unsqueeze(-1)
            else:
                start_token = last_spectrum

            tgt = torch.zeros_like(y_batch)
            tgt[:, 0] = start_token
            tgt[:, 1:] = y_batch[:, :-1]

            # Forward pass
            y_pred = model(src, tgt)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    # Concatenate batches
    y_pred_all = torch.cat(all_preds, dim=0)  # (num_samples, lead_time, output_dim)
    y_true_all = torch.cat(all_targets, dim=0)

    # Metrics per horizon (lead_time)
    lead_time = y_pred_all.shape[1]

    rmse_per_horizon = []
    mape_per_horizon = []
    cc_per_horizon = []

    for t in range(lead_time):
        y_pred_t = y_pred_all[:, t]
        y_true_t = y_true_all[:, t]

        rmse_per_horizon.append(rmse_fn(y_pred_t, y_true_t).item())
        mape_per_horizon.append(mean_absolute_percentage_error(y_pred_t, y_true_t).item())
        cc_per_horizon.append(pearson_corrcoef(y_pred_t, y_true_t).item())

    # Convert to tensors or lists
    metrics_per_horizon = {
        'RMSE': torch.tensor(rmse_per_horizon),
        'MAPE': torch.tensor(mape_per_horizon),
        'CC': torch.tensor(cc_per_horizon)
    }

    return metrics_per_horizon
