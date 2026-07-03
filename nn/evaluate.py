"""Evaluation utilities for the wave forecast transformer model."""
import numpy as np
import torch
from tqdm import tqdm
from torchmetrics.functional import pearson_corrcoef
from utils import RMSELoss, get_start_token, compute_bulk_params


def evaluate(model, dataloader, device='cpu', freqs=None, lead_time=None,
             freq_means=None):
    """
    Evaluate model using autoregressive inference.

    Persistence baseline: predict that every future step equals the last
    observed value at the end of the encoder window (i.e. the start token).
    This is constructed without any leakage — only information available at
    forecast time is used.

    Parameters:
    - model      : PyTorch model with an infer() method
    - dataloader : DataLoader yielding (X_batch, y_batch)
    - device     : 'cpu' or 'cuda'
    - freqs      : frequency tensor required for Hs computation and infer()
    - lead_time  : number of steps to forecast (passed to model.infer)
    - freq_means : torch.Tensor | None, shape (num_freqs,); per-frequency
                   training mean μ(f).  When provided, bulk-parameter metrics
                   (Hs, Tm02, Shape, SI) are computed after denormalising
                   E = Ẽ * μ(f), so results are in physical units.  Also
                   forwarded to model.infer() → get_start_token() for
                   physically correct Hs persistence baselines.  If None,
                   bulk metrics are skipped entirely.

    Returns dict with keys:
        'RMSE'               : overall RMSE (all steps, all samples flattened)
        'Hs_MAPE'            : MAPE (%) of Hs = 4√m₀, predicted vs target,
                               always in physical metres. For target == 'hs'
                               this is computed directly on y_pred/y_true
                               (already physical, per prepare_y). For
                               target == 'density' it is computed from Hs
                               derived out of the denormalised spectrum.
                               Hs is bounded away from zero for this buoy
                               (min ≈ 0.8 m observed), so unlike a per-bin
                               spectral MAPE this ratio never blows up.
                               Comparable to the "accuracy = 100% - MAPE"
                               metric reported in Hs-forecasting literature
                               (e.g. Minuzzi & Farina 2023, Londe & Panchang).
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

    When model.target == 'density' and freq_means is not None, five additional
    metrics are appended (all in physical units — m²/Hz spectra, metres for
    Hs, seconds for Tm02 — because predictions are denormalised before any
    integration):

    Bulk-parameter consistency (spectral moment integration):
        'Hs_RMSE'            : float, RMSE of Hs = 4√m₀  (predicted vs target)
        'Hs_Bias'            : float, mean signed error of Hs (predicted − target)
        'Tm02_RMSE'          : float, RMSE of Tm02 = √(m₀/m₂)
        'Tm02_Bias'          : float, mean signed error of Tm02 (predicted − target)

    Spectral shape error (energy magnitude removed; normalised by target m₀):
        'Shape_RMSE'         : float, mean per-spectrum RMSE of E(f)/m₀_target
                               (averaged over valid spectra, i.e. m₀_target ≥
                               M0_MASK_THRESHOLD; isolates shape errors from
                               energy magnitude errors)
        'Shape_masked_samples': int, number of (sample, step) pairs excluded
                               because m₀_target was below the threshold

    Per-frequency-bin Scatter Index:
        'SI_per_bin'         : list[float], length num_freqs; SI[i] =
                               RMSE(E_pred[:,i], E_target[:,i]) /
                               mean(E_target[:,i]), computed over all
                               (sample × step) pairs flattened together;
                               reveals which parts of the spectrum are hardest
                               to forecast
        'SI_mean'            : float, mean of SI_per_bin across all bins;
                               scalar summary for monitoring

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
            y_pred = model.infer(src, freqs, lead_time, freq_means=freq_means)

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
    cc         = pearson_corrcoef(y_pred_flat, y_true_flat).item()
    overall_ss = 1.0 - rmse / rmse_pers if rmse_pers > 0 else float('nan')
    bias       = (y_pred_flat - y_true_flat).mean().item()
    ss_res     = ((y_true_flat - y_pred_flat) ** 2).sum()
    ss_tot     = ((y_true_flat - y_true_flat.mean()) ** 2).sum()
    r2         = (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')

    # Hs_MAPE: for target == 'hs', y_pred/y_true ARE physical Hs already
    # (prepare_y computes compute_hs on the raw, non-normalised density
    # before any scaling is applied). For target == 'density' this is
    # overwritten below once Hs is derived from the denormalised spectrum.
    hs_mape = float(100.0 * torch.mean(
        torch.abs(y_pred_flat - y_true_flat) / torch.abs(y_true_flat)
    ).item()) if model.target == 'hs' else None

    # Density-target-only metrics — bulk parameters, spectral shape, and SI.
    # All integrations must operate on physical spectral density E (m² Hz⁻¹),
    # not on the normalised form Ẽ = E / μ(f) stored in the dataloader.
    # Denormalisation is E = Ẽ * μ(f) applied once here before every metric.
    #
    # M0_MASK_THRESHOLD is in physical units (m²): (sample, step) pairs whose
    # target m₀ is below this value are excluded from Shape_RMSE to avoid
    # dividing a near-zero denominator.  Typical open-ocean m₀ is O(0.01–1 m²);
    # 1e-4 m² (Hs ≈ 4 cm) excludes only genuinely calm or missing records.
    M0_MASK_THRESHOLD = 1e-4   # m²

    bulk = {}
    if model.target == 'density' and freqs is not None and freq_means is not None:
        freqs_np = freqs.cpu().numpy()                        # (num_freqs,)
        fm_np    = freq_means.cpu().numpy()                   # (num_freqs,)

        # Denormalise: E_phys = Ẽ * μ(f);  shape (batch, lead_time, num_freqs)
        pred_np = y_pred_all.numpy() * fm_np[np.newaxis, np.newaxis, :]
        true_np = y_true_all.numpy() * fm_np[np.newaxis, np.newaxis, :]

        # --- Bulk parameter consistency ---
        # compute_bulk_params uses torch.trapezoid-equivalent trapezoidal
        # integration over the actual (log-spaced) frequency grid.
        hs_pred, tm02_pred = compute_bulk_params(pred_np, freqs_np)
        hs_true, tm02_true = compute_bulk_params(true_np, freqs_np)

        hs_err   = hs_pred   - hs_true   # (batch, lead_time), metres
        tm02_err = tm02_pred - tm02_true  # (batch, lead_time), seconds
        # Hs is bounded away from zero for this buoy (min ~0.8 m observed),
        # so this ratio is well-behaved, unlike a per-bin spectral MAPE.
        hs_mape  = float(100.0 * np.mean(np.abs(hs_err) / np.abs(hs_true)))

        # --- Spectral shape error ---
        # Normalise both pred and true by the TARGET physical m₀ so that shape
        # errors are decoupled from energy magnitude errors.
        m0_true = np.trapezoid(true_np, freqs_np, axis=2)   # (batch, lead_time), m²
        valid    = m0_true >= M0_MASK_THRESHOLD               # (batch, lead_time) bool
        n_masked = int((~valid).sum())

        if valid.any():
            # Replace masked entries with 1.0 before dividing to avoid NaN;
            # those rows are excluded from the final mean via boolean indexing.
            m0_denom = np.where(valid, m0_true, 1.0)[:, :, np.newaxis]
            shape_pred = pred_np / m0_denom   # (batch, lead_time, num_freqs)
            shape_true = true_np / m0_denom

            per_spectrum_rmse = np.sqrt(
                ((shape_pred - shape_true) ** 2).mean(axis=2)   # (batch, lead_time)
            )
            shape_rmse = float(per_spectrum_rmse[valid].mean())
        else:
            shape_rmse = float('nan')

        # --- Per-frequency-bin Scatter Index ---
        # Flatten to (N, num_freqs) so each bin's RMSE and mean are computed
        # over all (sample × step) pairs.  SI[i] = RMSE_i / mean(E_true_i).
        flat_pred = pred_np.reshape(-1, pred_np.shape[2])   # (N, num_freqs)
        flat_true = true_np.reshape(-1, true_np.shape[2])

        rmse_per_bin = np.sqrt(((flat_pred - flat_true) ** 2).mean(axis=0))  # (num_freqs,)
        mean_per_bin = flat_true.mean(axis=0).clip(min=1e-12)                # (num_freqs,)
        si_per_bin   = rmse_per_bin / mean_per_bin                           # (num_freqs,)

        bulk = {
            'Hs_RMSE'             : float(np.sqrt(np.mean(hs_err   ** 2))),
            'Hs_Bias'             : float(np.mean(hs_err)),
            'Hs_MAPE'             : hs_mape,
            'Tm02_RMSE'           : float(np.sqrt(np.mean(tm02_err ** 2))),
            'Tm02_Bias'           : float(np.mean(tm02_err)),
            'Shape_RMSE'          : shape_rmse,
            'Shape_masked_samples': n_masked,
            'SI_per_bin'          : si_per_bin.tolist(),
            'SI_mean'             : float(si_per_bin.mean()),
        }

    return {
        'RMSE'               : rmse,
        'Hs_MAPE'            : hs_mape,
        'CC'                 : cc,
        'Bias'               : bias,
        'R2'                 : r2,
        'per_step_RMSE'      : per_step_rmse,
        'per_step_RMSE_pers' : per_step_rmse_pers,
        'per_step_SS'        : per_step_ss,
        'per_step_Bias'      : per_step_bias,
        'per_step_R2'        : per_step_r2,
        'overall_SS'         : overall_ss,
        **bulk,
    }
