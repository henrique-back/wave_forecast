from utils import get_start_token, RMSELoss
import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, device='cpu', freqs=None):
    model.train()
    total_loss = 0.0
    loss_fn = RMSELoss()

    loop = tqdm(dataloader, desc='Training', leave=False)

    for src, y_batch in loop:
        src = src.to(device)  # Encoder input
        y_batch = y_batch.to(device)  # Ground truth future sequence
        
        if model.target == 'hs' and y_batch.dim() == 2:
            y_batch = y_batch.unsqueeze(-1)
            
        # Initialize tgt tensor, same shape as y_batch
        tgt = torch.zeros_like(y_batch).to(device)

        start_token = get_start_token(src, model.target, freqs, device)
        tgt[:, 0, :] = start_token

        # Teacher forcing: feed full target shifted right by 1 as decoder input
        tgt[:, 1:, :] = y_batch[:, :-1, :]

        # Forward pass
        y_pred = model(src, tgt)

        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * src.size(0)
        
        loop.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)
    
    return {'RMSE': avg_loss}
