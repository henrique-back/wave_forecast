from utils import (get_start_token, RMSELoss, trapz_weights, to_log_space,
                   SpectralWassersteinLoss, SpectralKLDivergenceLoss,
                   SoftPeakHeightLoss, find_peak_windows)
import numpy as np
import torch
from tqdm import tqdm


def _peak_windows_for_batch(y_batch, freqs_np, max_peaks=4, f_max=0.4,
                             energy_frac=0.05, min_bins=2):
    """Per-batch, per-step peak-window detection for SoftPeakHeightLoss.

    Pragmatic alternative to precomputing windows once at data-preparation
    time (see utils.loss.SoftPeakHeightLoss's docstring) — accepted for now
    since this only runs when peak_loss_weight > 0. See
    manuscript/decisions/log/025.

    Parameters
    ----------
    y_batch : torch.Tensor, shape (batch, lead_time, num_freqs)
        LOG-space true spectrum (already floored/converted — see
        train_one_epoch). Exponentiated internally.
    freqs_np : np.ndarray, shape (num_freqs,)
    max_peaks : int
        Fixed padding width for the peak axis — a (sample, step) with
        fewer real peaks gets padding slots (peak_mask=False there,
        SoftPeakHeightLoss excludes them); one with MORE than max_peaks
        significant peaks (rare — see find_significant_peaks' Portilla
        criteria) has its highest-frequency extras dropped (windows come
        back in ascending-frequency order from find_peak_windows).
    f_max, energy_frac, min_bins : forwarded to find_peak_windows.

    Returns
    -------
    left_idx, right_idx : torch.LongTensor, shape (batch, lead_time, max_peaks)
    peak_mask : torch.BoolTensor, shape (batch, lead_time, max_peaks)
    """
    batch, lead_time, _ = y_batch.shape
    true_phys = torch.exp(y_batch).detach().cpu().numpy()

    left_idx = np.zeros((batch, lead_time, max_peaks), dtype=np.int64)
    right_idx = np.zeros((batch, lead_time, max_peaks), dtype=np.int64)
    peak_mask = np.zeros((batch, lead_time, max_peaks), dtype=bool)

    for b in range(batch):
        for t in range(lead_time):
            windows = find_peak_windows(freqs_np, true_phys[b, t], f_max, energy_frac, min_bins)
            for k, (_, l, r) in enumerate(windows[:max_peaks]):
                left_idx[b, t, k] = l
                right_idx[b, t, k] = r
                peak_mask[b, t, k] = True

    device = y_batch.device
    return (torch.from_numpy(left_idx).to(device),
            torch.from_numpy(right_idx).to(device),
            torch.from_numpy(peak_mask).to(device))


def train_one_epoch(model, dataloader, optimizer, device='cpu', freqs=None,
                    tf_ratio=1.0, freq_means=None, shape_means=None,
                    wasserstein_loss_weight=0.0, kl_loss_weight=0.0,
                    base_loss_weight=1.0, peak_loss_weight=0.0, peak_max_count=4):
    """Train for one epoch and return {'RMSE': avg_loss}.

    avg_loss is the mean per-sample training loss actually optimised: RMSE
    for 'hs', but plain MSE in log-space for 'density'/'shape' (see the
    loss computation below) — the dict key is kept as 'RMSE' for logging/
    call-site compatibility, but for density/shape this value is not on the
    same scale as pre-ablation runs or as evaluate()'s reported 'RMSE'.

    Parameters
    ----------
    freq_means : torch.Tensor | None, shape (num_freqs,)
        Per-frequency training mean μ(f) of the physical density. When
        provided:
        - For 'hs' target      : passed to get_start_token so the decoder
          start token is in physical metres (E = Ẽ * μ(f) before
          integration).
        - For 'density' target : y_batch is converted to log-spectral-energy
          — log(Ẽ * μ(f)), floored per utils.to_log_space — immediately
          after load, before it's used to build the decoder input or the
          loss target. The model now predicts this log-space quantity
          directly, so both the teacher-forced decoder input and the
          scheduled-sampling self-feedback loop below operate consistently
          in log-space with no further special-casing.
    shape_means : torch.Tensor | None, shape (num_freqs,)
        Per-frequency training mean of the physical unit-area shape target.
        Required for target == 'shape': y_batch (already physical, per
        prepare_y) is converted to log-space the same way as above.
    wasserstein_loss_weight : float
        target in ('density', 'shape') only, default 0.0. When > 0, adds
        wasserstein_loss_weight * utils.SpectralWassersteinLoss(y_pred,
        y_batch, freqs) to the main per-bin loss. See utils/loss.py and
        manuscript/decisions/log/020, 024.
    kl_loss_weight : float
        target in ('density', 'shape') only, default 0.0. When > 0, adds
        kl_loss_weight * utils.SpectralKLDivergenceLoss(y_pred, y_batch,
        freqs) to the main per-bin loss. Complementary to (not a substitute
        for) the Wasserstein term — see utils/loss.py's docstring.
    base_loss_weight : float, default 1.0
        Multiplies the main per-bin loss term before any auxiliary term is
        added. Set to 0.0 by the loss-ablation study to literally SUBSTITUTE
        the per-bin loss with the auxiliary terms rather than adding to it.
        See manuscript/decisions/log/026.
    peak_loss_weight : float
        target in ('density', 'shape') only, default 0.0. When > 0, adds
        peak_loss_weight * utils.SoftPeakHeightLoss(y_pred, y_batch, freqs,
        left_idx, right_idx, peak_mask) to the loss. Skipped (no
        contribution) for any batch where SoftPeakHeightLoss's 'mean'
        reduction returns NaN (no sample in that batch has any detected
        peak) rather than poisoning the batch's gradient with NaN.
    peak_max_count : int, default 4
        Forwarded to _peak_windows_for_batch as max_peaks. No-op when
        peak_loss_weight == 0.

    For 'density'/'shape' targets, the loss is additionally weighted across
    the frequency axis by utils.trapz_weights(freqs) — the grid is
    log-spaced (dense near 0.02 Hz, coarse near 0.485 Hz), so a flat
    elementwise MSE over-weights the dense low-frequency region relative to
    its actual share of the physical spectrum. 'hs' has no frequency axis
    (output_dim=1) so it's unaffected.
    """
    model.train()
    total_loss = 0.0
    loss_fn = RMSELoss()
    wasserstein_loss_fn = SpectralWassersteinLoss()
    kl_loss_fn = SpectralKLDivergenceLoss()
    peak_loss_fn = SoftPeakHeightLoss()

    freq_weights = None
    freqs_np = None
    if model.target in ('density', 'shape') and freqs is not None:
        freqs_np = freqs.cpu().numpy()
        freq_weights = torch.from_numpy(
            trapz_weights(freqs_np)
        ).to(device=device, dtype=torch.float32)

    loop = tqdm(dataloader, desc='Training', leave=False)

    for src, aux, y_batch in loop:
        src = src.to(device)  # Encoder input
        aux = aux.to(device)  # Auxiliary encoder side-input (e.g. wind), may be zero-width
        y_batch = y_batch.to(device)  # Ground truth future sequence

        if model.target == 'hs' and y_batch.dim() == 2:
            y_batch = y_batch.unsqueeze(-1)

        # Convert y_batch to log-spectral-energy space immediately after
        # load, before it's used anywhere downstream (decoder input
        # construction, scheduled sampling, loss) — see docstring above.
        if model.target == 'density':
            if freq_means is None:
                raise ValueError("freq_means is required for target='density'")
            fm = freq_means.to(device)
            y_batch = to_log_space(y_batch * fm, fm)
        elif model.target == 'shape':
            if shape_means is None:
                raise ValueError("shape_means is required for target='shape'")
            y_batch = to_log_space(y_batch, shape_means.to(device))

        start_token = get_start_token(src, model.target, freqs, device,
                                      freq_means=freq_means, shape_means=shape_means)

        if tf_ratio >= 1.0:
            # Pure teacher forcing: decoder always receives the ground-truth
            # previous step (fast single forward pass).
            tgt = torch.zeros_like(y_batch).to(device)
            tgt[:, 0, :] = start_token
            tgt[:, 1:, :] = y_batch[:, :-1, :]
            y_pred = model(src, tgt, aux=aux)
        else:
            # Scheduled sampling: for each sample in the batch independently,
            # feed the ground-truth previous token with probability tf_ratio,
            # and the model's own previous prediction with probability
            # (1 - tf_ratio). Drawn per-sample, not per-batch — see
            # manuscript/decisions/log/012.
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
                    use_teacher = torch.rand(y_batch.size(0), 1, 1, device=device) < tf_ratio
                    teacher_token = y_batch[:, t:t+1, :]
                    model_token = pred_t.detach()
                    next_input = torch.where(use_teacher, teacher_token, model_token)
                    decoder_input = torch.cat([decoder_input, next_input], dim=1)

            y_pred = torch.cat(all_preds, dim=1)  # (batch, lead_time, output_dim)

        # 'density'/'shape' targets: plain MSE (no sqrt) directly on y_pred
        # vs y_batch, both in log-space — y_batch was already converted to
        # log-spectral-energy above, and the model predicts that same
        # log-space quantity directly, so no further denormalisation is
        # needed here. 'hs': unchanged RMSE on physical metres (see
        # prepare_y and get_start_token).
        squared = model.target in ('density', 'shape')
        loss = base_loss_weight * loss_fn(y_pred, y_batch, weights=freq_weights, squared=squared)

        if model.target in ('density', 'shape') and wasserstein_loss_weight > 0:
            loss = loss + wasserstein_loss_weight * wasserstein_loss_fn(y_pred, y_batch, freqs)

        if model.target in ('density', 'shape') and kl_loss_weight > 0:
            loss = loss + kl_loss_weight * kl_loss_fn(y_pred, y_batch, freqs)

        if model.target in ('density', 'shape') and peak_loss_weight > 0:
            left_idx, right_idx, peak_mask = _peak_windows_for_batch(
                y_batch, freqs_np, max_peaks=peak_max_count)
            peak_term = peak_loss_fn(y_pred, y_batch, freqs, left_idx, right_idx, peak_mask)
            if not torch.isnan(peak_term):
                loss = loss + peak_loss_weight * peak_term

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
