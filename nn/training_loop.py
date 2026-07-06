from utils import get_start_token, RMSELoss
import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, device='cpu', freqs=None,
                    tf_ratio=1.0, freq_means=None):
    """Train for one epoch and return {'RMSE': avg_loss}.

    Parameters
    ----------
    freq_means : torch.Tensor | None, shape (num_freqs,)
        Per-frequency training mean μ(f).  When provided:
        - For 'hs' target  : passed to get_start_token so the decoder start
          token is in physical metres (E = Ẽ * μ(f) before integration).
        - For 'density' target : loss is computed in physical space by
          denormalising both prediction and target before RMSE, i.e.
          loss = RMSE(ŷ * μ(f), y * μ(f)).  This prevents the loss from
          being dominated by the normalisation artefact.
        If None, falls back to normalised-space loss (old behaviour).
    """
    model.train()
    total_loss = 0.0
    loss_fn = RMSELoss()

    loop = tqdm(dataloader, desc='Training', leave=False)

    for src, aux, y_batch in loop:
        src = src.to(device)  # Encoder input
        aux = aux.to(device)  # Auxiliary encoder side-input (e.g. wind), may be zero-width
        y_batch = y_batch.to(device)  # Ground truth future sequence

        if model.target == 'hs' and y_batch.dim() == 2:
            y_batch = y_batch.unsqueeze(-1)

        start_token = get_start_token(src, model.target, freqs, device,
                                      freq_means=freq_means)

        if tf_ratio >= 1.0:
            # Pure teacher forcing: decoder always receives the ground-truth
            # previous step (fast single forward pass).
            tgt = torch.zeros_like(y_batch).to(device)
            tgt[:, 0, :] = start_token
            tgt[:, 1:, :] = y_batch[:, :-1, :]
            y_pred = model(src, tgt, aux=aux)
        else:
            # Scheduled sampling: at each step, feed the ground-truth previous
            # token with probability tf_ratio, and the model's own previous
            # prediction with probability (1 - tf_ratio).  This closes the gap
            # between teacher-forced training and autoregressive evaluation.
            lead_time = y_batch.shape[1]
            # src never changes across decode steps — encode it once and
            # reuse across the loop instead of re-running the encoder at
            # every step (see WaveHeightBaselineNN.encode/decode).
            memory = model.encode(src, aux=aux)
            decoder_input = start_token.unsqueeze(1)  # (batch, 1, output_dim)
            all_preds = []

            for t in range(lead_time):
                preds = model.decode(decoder_input, memory)  # (batch, t+1, output_dim)
                pred_t = preds[:, -1:, :]             # (batch, 1, output_dim)
                all_preds.append(pred_t)

                if t < lead_time - 1:
                    if torch.rand(1).item() < tf_ratio:
                        next_input = y_batch[:, t:t+1, :]   # teacher token
                    else:
                        next_input = pred_t.detach()         # model's own token
                    decoder_input = torch.cat([decoder_input, next_input], dim=1)

            y_pred = torch.cat(all_preds, dim=1)  # (batch, lead_time, output_dim)

        # Compute loss in physical space for the density target:
        # denormalise E = Ẽ * μ(f) before RMSE so that the gradient signal
        # reflects actual spectral energy magnitudes (m² Hz⁻¹), not
        # normalised units.  For the Hs target the predictions and targets
        # are already in physical metres (see prepare_y and get_start_token).
        if model.target == 'density' and freq_means is not None:
            fm = freq_means.to(device)          # (num_freqs,)
            loss = loss_fn(y_pred * fm, y_batch * fm)
        else:
            loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent early-training spikes from pushing parameters
        # into flat loss-surface regions.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * src.size(0)

        loop.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader.dataset)

    return {'RMSE': avg_loss}
