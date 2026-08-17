from utils import (get_start_token, RMSELoss, trapz_weights, to_log_space,
                   SpectralWassersteinLoss, SpectralKLDivergenceLoss)
import torch
from tqdm import tqdm


def train_one_epoch(model, dataloader, optimizer, device='cpu', freqs=None,
                    tf_ratio=1.0, freq_means=None, shape_means=None,
                    wasserstein_loss_weight=0.0, kl_loss_weight=0.0):
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
        target in ('density', 'shape') only. Default 0.0 (no behavior
        change). When > 0, adds wasserstein_loss_weight *
        utils.SpectralWassersteinLoss(y_pred, y_batch, freqs) to the main
        per-bin loss — the 1-D earth-mover distance between predicted and
        true spectra (exact via CDF L1 distance, see utils/loss.py).
        SpectralWassersteinLoss internally exp()s and mass-normalizes its
        input, so it is not actually shape-specific — the same call works
        for 'density's log-spectral-energy y_pred/y_batch unchanged. Unlike
        the main per-bin loss, W1 is forgiving of small peak-position shifts
        while still penalizing a blurred/flattened prediction relative to a
        sharp true spectrum — aimed at the same multimodal-blur problem as
        the (reverted) SpectralSlopeLoss experiment, via a different
        mechanism.
    kl_loss_weight : float
        target in ('density', 'shape') only. Default 0.0 (no behavior
        change). When > 0, adds kl_loss_weight *
        utils.SpectralKLDivergenceLoss(y_pred, y_batch, freqs) to the main
        per-bin loss — KL divergence between predicted/true spectra treated
        as Δf-weighted probability distributions over frequency (see
        utils/loss.py; gradient-equivalent to a plain cross-entropy term,
        used instead of one purely so this reports exactly 0 at a perfect
        match). Unlike wasserstein_loss_weight, NOT yet tuned by
        nn/optimization.py::objective() — no manually-swept range exists
        yet to base a search bracket on, so this is a manual-A/B-only
        parameter for now (Stage 1); promoting it to Optuna's search space
        is a deferred follow-up. Complementary to (not a substitute for)
        the Wasserstein term: this KL term has no cross-bin spatial
        awareness (a coherently shifted peak is still fully penalized), but
        weights the main per-bin loss's floor-crossing errors by how much
        true probability mass the affected bin actually holds, discounting
        a peak that broadens into its neighbourhood (while still covering
        its true bin) relative to the current frequency-weighted
        log-space MSE, which penalizes that case more than a full shift.

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

    freq_weights = None
    if model.target in ('density', 'shape') and freqs is not None:
        freq_weights = torch.from_numpy(
            trapz_weights(freqs.cpu().numpy())
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
            # (1 - tf_ratio). This closes the gap between teacher-forced
            # training and autoregressive evaluation.
            #
            # The choice is drawn per-sample (not once for the whole batch)
            # so that, on average, every batch/epoch sees a mix of teacher-
            # forced and self-generated context close to the target tf_ratio.
            # A single batch-wide draw would instead make whole batches
            # uniformly "easy" (teacher-forced) or "hard" (autoregressive,
            # error-compounding) purely by chance, injecting a lot of
            # spurious epoch-to-epoch variance into the training signal.
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
        loss = loss_fn(y_pred, y_batch, weights=freq_weights, squared=squared)

        if model.target in ('density', 'shape') and wasserstein_loss_weight > 0:
            loss = loss + wasserstein_loss_weight * wasserstein_loss_fn(y_pred, y_batch, freqs)

        if model.target in ('density', 'shape') and kl_loss_weight > 0:
            loss = loss + kl_loss_weight * kl_loss_fn(y_pred, y_batch, freqs)

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
