import torch
import torch
from torchmetrics.functional import (
    mean_squared_error,
    mean_absolute_percentage_error,
    pearson_corrcoef
)

def evaluate(model, dataloader, loss_fn, device='cpu', freqs=None, **loss_kwargs):
    """
    Evaluate model with MSE, MAPE, Pearson CC.

    Parameters:
    - model: PyTorch model
    - dataloader: DataLoader yielding (X_batch, y_batch)
    - loss_fn: loss function (e.g., nn.MSELoss() or custom callable)
    - device: computation device ('cpu' or 'cuda')
    - freqs: frequencies needed for start token (tensor or numpy array)
    - **loss_kwargs: additional arguments passed to loss_fn (if needed)

    Returns:
    - MSE: 
    - MAPE:
    - Pearson CC:
    float, average loss over the dataset
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for src, y_batch in dataloader:
            src = src.to(device)
            y_batch = y_batch.to(device)

            # Prepare decoder input
            last_spectrum = src[:, -1, :, 0]  # Select spectrum density channel

            if model.target == 'hs':
                hs = compute_hs(last_spectrum.cpu().numpy(), freqs.cpu().numpy())
                hs = torch.from_numpy(hs).to(device).float()
                start_token = hs.unsqueeze(-1)
            else:
                start_token = last_spectrum

            # Prepare full decoder input with teacher forcing
            tgt = torch.zeros_like(y_batch)
            tgt[:, 0] = start_token
            tgt[:, 1:] = y_batch[:, :-1]

            # Forward pass
            y_pred = model(src, tgt)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    # Concatenate all batches
    y_pred_all = torch.cat(all_preds, dim=0).flatten()
    y_true_all = torch.cat(all_targets, dim=0).flatten()

    # Compute metrics
    mse = mean_squared_error(y_pred_all, y_true_all).item()
    mape = mean_absolute_percentage_error(y_pred_all, y_true_all).item()
    cc = pearson_corrcoef(y_pred_all, y_true_all).item()

    return {'MSE': mse, 'MAPE': mape, 'CC': cc}