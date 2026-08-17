"""Evaluation utilities for the wave forecast transformer model."""
import numpy as np
import torch
from tqdm import tqdm
from torchmetrics.functional import pearson_corrcoef
from utils import (RMSELoss, get_start_token, compute_bulk_params, trapz_weights,
                   to_log_space, peak_modality_metrics, SpectralWassersteinLoss)


def evaluate(model, dataloader, device='cpu', freqs=None, lead_time=None,
             freq_means=None, shape_means=None, return_arrays=False,
             compute_peak_metrics=False):
    """
    Evaluate model using autoregressive inference.

    NOTE on log-space (density/shape targets): y_batch, model.infer()'s
    output, and the persistence baseline are all in log-spectral-energy
    space for 'density' (log E(f)) / 'shape' (log E(f)/m0) targets — see
    nn/training_loop.py for why this must be applied to the entire y_batch
    tensor, not just at loss time. Consequently every metric below that
    operates directly on y_pred_all/y_true_all/y_pers_all without an
    explicit exp() — i.e. the top-level 'RMSE', 'CC', 'Bias', 'R2',
    'per_step_*', 'overall_SS' — is computed in log-space for these two
    targets, and is NOT comparable to pre-ablation runs (where these were
    physical/normalised-space). The bulk-parameter block below (Hs/Tm02/
    Shape/SI) explicitly exp()'s back to physical units and remains
    comparable across the ablation.

    Persistence baseline: predict that every future step equals the last
    observed value at the end of the encoder window (i.e. the start token).
    This is constructed without any leakage — only information available at
    forecast time is used.

    Parameters:
    - model      : PyTorch model with an infer() method
    - dataloader : DataLoader yielding (X_batch, aux_batch, y_batch)
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
    - return_arrays : bool, if True also return the raw
                   (y_pred_all, y_true_all, y_pers_all) tensors — shape
                   (total_samples, lead_time, output_dim) — alongside the
                   metrics dict, as (metrics, arrays). Lets callers (e.g. an
                   aggregate test-set plot) reuse this function's single
                   autoregressive inference pass instead of re-running it.
    - compute_peak_metrics : bool, target == 'shape' only. Default False —
                   this function is called every epoch of every Optuna trial
                   (nn/optimization.py) plus once per test split per trial/
                   seed, so the extra per-sample peak-detection pass (see
                   'Peak-aware multimodal metrics' below) is opt-in rather
                   than unconditional, to keep the hyperparameter search's
                   hot path at zero added cost. Set True for one-shot
                   post-hoc analysis (scripts/infer.py --save-metrics).

    Returns dict with keys:
        'RMSE'               : overall RMSE (all steps, all samples). For
                               'density'/'shape' targets this is a
                               frequency-weighted (trapezoidal) RMSE across
                               the frequency axis — see note below; 'hs' has
                               no frequency axis and is unaffected.
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
        'CC'                 : Pearson correlation (frequency-weighted for
                               'density'/'shape' — see note below)
        'Bias'               : mean error = mean(pred - true) (frequency-
                               weighted for 'density'/'shape')
        'R2'                 : coefficient of determination (frequency-
                               weighted for 'density'/'shape')
        'per_step_RMSE'      : list[float], one RMSE per forecast step
        'per_step_SS'        : list[float], Skill Score vs persistence per step
                               SS = 1 - RMSE_model / RMSE_persistence
                               SS > 0: model beats persistence
                               SS = 1: perfect forecast
                               SS < 0: persistence is better
        'per_step_Bias'      : list[float], mean error per forecast step
        'per_step_R2'        : list[float], R² per forecast step
        'overall_SS'         : float, Skill Score computed on overall RMSE

    Always present:
        'Hs_SS'              : float, Skill Score for significant wave height.
                               For target=='hs': equals per_step_SS[-1] on the
                               physical Hs predictions — the final forecast
                               step (the actual chosen lead time), not an
                               average over the autoregressive steps leading
                               up to it. For target=='density': 1 − Hs_RMSE_model /
                               Hs_RMSE_persistence, pooled across all steps
                               (computed from denormalised spectra — see the
                               density-target block below, which is NOT
                               final-step-only). Robust to seq_len variation;
                               suitable as the Optuna objective when Hs
                               accuracy is the primary goal.

    When model.target == 'density' and freq_means is not None, five additional
    metrics are appended (all in physical units — m²/Hz spectra, metres for
    Hs, seconds for Tm02 — because predictions are denormalised before any
    integration). All five are computed on the FINAL forecast step only (the
    actual chosen lead time), not pooled across the autoregressive steps
    leading up to it — same reasoning as the 'hs'-target Hs_SS and
    'final_step_SS' in nn/optimization.py::_compute_val_score:

    Bulk-parameter consistency (spectral moment integration):
        'Hs_RMSE'            : float, RMSE of Hs = 4√m₀  (predicted vs target)
                               at the final step
        'Hs_Bias'            : float, mean signed error of Hs (predicted −
                               target) at the final step
        'Tm02_RMSE'          : float, RMSE of Tm02 = √(m₀/m₂) at the final step
        'Tm02_Bias'          : float, mean signed error of Tm02 (predicted −
                               target) at the final step

    Spectral shape error (energy magnitude removed; normalised by target m₀):
        'Shape_RMSE'         : float, mean per-spectrum RMSE of E(f)/m₀_target
                               at the final step (averaged over valid spectra,
                               i.e. m₀_target ≥ M0_MASK_THRESHOLD; isolates
                               shape errors from energy magnitude errors). The
                               per-spectrum RMSE itself is a frequency-weighted
                               (trapezoidal) mean across bins — see note below.
        'Shape_masked_samples': int, number of samples (at the final step)
                               excluded because m₀_target was below the threshold
        'Shape_SS'           : float, Skill Score of Shape_RMSE vs a
                               persistence baseline computed the same way
                               (last observed shape held constant), at the
                               final step.
        'Shape_Wasserstein'  : float, 1-D Wasserstein (earth-mover) distance
                               in the same mass-normalized shape space as
                               Shape_RMSE/SS, masked the same way (near-zero
                               m₀_target samples excluded) — see
                               utils.SpectralWassersteinLoss and the 'shape'
                               target's own 'Shape_Wasserstein' entry below
                               (same metric name, same underlying class, used
                               for both target types).

    Per-frequency-bin Scatter Index (final step only):
        'SI_per_bin'         : list[float], length num_freqs; SI[i] =
                               RMSE(E_pred[:,i], E_target[:,i]) /
                               mean(E_target[:,i]), computed over all samples
                               at the final forecast step; reveals which parts
                               of the spectrum are hardest to forecast
        'SI_mean'            : float, frequency-weighted (trapezoidal) mean of
                               SI_per_bin across all bins; scalar summary for
                               monitoring

    When model.target == 'shape' and freqs is not None, additional metrics
    are appended. Unlike the 'density' block above, these do NOT require
    freq_means (predictions/targets are already the unit-area shape, per
    nn/prepare_y.py) and are NOT masked by M0_MASK_THRESHOLD — prepare_y
    discards m₀ for the 'shape' target, so degenerate low-energy spectra
    can't be identified/excluded here the way the 'density' branch does:
        'per_step_RMSE_phys' : list[float], one per forecast step — the SAME
                               per-step RMSE as the top-level 'per_step_RMSE'
                               list, but computed in physical (exp()'d)
                               unit-area shape space rather than log-shape
                               space. The top-level 'per_step_*' keys stay in
                               log-space deliberately (matches the training
                               loss — see docstring NOTE above — so Optuna/
                               early-stopping/LR-scheduler behavior is
                               unaffected), which makes them NOT comparable
                               across a lead-time curve to any always-physical
                               source (e.g. utils/linear_baseline.py, or a
                               'density'-target model). Use these '_phys' keys
                               instead whenever a per-step trend needs to be
                               plotted or diffed against a physical baseline
                               (see scripts/compare_versions.py).
        'per_step_RMSE_pers_phys', 'per_step_SS_phys', 'per_step_Bias_phys' :
                               same physical-space treatment, for the
                               persistence RMSE / Skill Score / bias curves.
        'RMSE_per_bin', 'RMSE_per_bin_pers', 'Bias_per_bin', 'SI_per_bin' :
                               list[float], length num_freqs; the FINAL
                               forecast step's error broken out per
                               frequency bin instead of collapsed across the
                               spectrum — physical unit-area shape space,
                               same formula as the 'density' target's
                               'SI_per_bin' above (SI[i] = RMSE_i /
                               mean(E_true_i)), just without the
                               M0_MASK_THRESHOLD masking (not needed here —
                               see the block's own comment). '_pers' is the
                               persistence baseline's per-bin RMSE, for a
                               reference curve alongside the model's.
        'Shape_RMSE'         : float, equal to 'per_step_RMSE_phys[-1]' — the final
                               forecast step's frequency-weighted shape RMSE,
                               NOT the all-step-pooled top-level 'RMSE' —
                               exposed under this name so it's directly
                               comparable to the 'density' target's Shape_RMSE
                               (also final-step-only).
        'Shape_SS'           : float, equal to 'per_step_SS_phys[-1]' — the final
                               step's shape-space Skill Score vs the
                               last-observed-shape persistence baseline, NOT
                               the all-step-pooled top-level 'overall_SS'.
        'Shape_Wasserstein'  : float, equal to 'per_step_Wasserstein[-1]' —
                               1-D Wasserstein-2 (quadratic earth-mover)
                               distance between predicted and true
                               final-step shape — see
                               utils.SpectralWassersteinLoss. Computed via
                               quantile-function inversion (no OT solver/
                               critic network needed for a 1-D distribution,
                               but — unlike Wasserstein-1's exact CDF-L1
                               shortcut — genuinely needs the quantile
                               domain; see that class's docstring). Each
                               spectrum is normalized by its own mass before
                               comparison, so this isolates shape error from
                               magnitude error (already tracked separately
                               by 'Shape_Mass_Error' below). Unlike
                               'Shape_RMSE', forgiving of small peak-position
                               shifts while still penalizing a blurred/
                               flattened prediction — complementary read on
                               the same multimodal-blur question the peak
                               metrics below address. Cheap pure tensor math,
                               so always computed (no opt-in flag, unlike the
                               peak metrics' scipy loop below).
        'per_step_Wasserstein', 'per_step_Wasserstein_pers' : list[float],
                               one per forecast step — same model/persistence
                               pairing as 'per_step_RMSE_phys'/
                               'per_step_RMSE_pers_phys', but the shift-
                               tolerant Wasserstein distance instead of RMSE.
                               Already physical (SpectralWassersteinLoss
                               exponentiates internally), no '_phys' suffix
                               needed — there is no log-space variant of this
                               metric to begin with.
        'Wasserstein_per_bin', 'Wasserstein_per_bin_pers' : list[float],
                               length num_freqs; the FINAL step's Wasserstein
                               distance broken out per (true spectrum's own)
                               frequency bin — mean over samples of
                               (F_pred^{-1}(cdf_true(f)) - f)^2, the SQUARED
                               quantile-domain gap BEFORE the final
                               integration-over-cumulative-probability — see
                               SpectralWassersteinLoss's reduction='per_bin'.
                               NOT integrable against `freqs` to recover
                               'Shape_Wasserstein' (per_bin's domain is
                               cumulative probability, non-uniformly spaced
                               in frequency — trapz(this, freqs) is not a
                               meaningful quantity; integrate against the
                               corresponding cdf_true instead, as
                               SpectralWassersteinLoss.forward does
                               internally). Deliberately NOT the same shape
                               as 'RMSE_per_bin': a peak shifted by one bin
                               shows up here as a bump straddling the true
                               peak, not a spike exactly at it, since this
                               tracks displaced probability MASS rather than
                               a pointwise value gap — read the two together
                               to tell "peak shifted" from "peak missed".
        'Shape_Mass_Error'   : float, mean absolute deviation of
                               ∫S_pred(f) df from 1 across all (sample, step)
                               pairs — deliberately NOT final-step-only, since
                               it's a renormalization sanity check (catch a
                               regression at ANY step), not a lead-time
                               performance metric. model.infer() already
                               renormalizes each predicted step to unit area,
                               so this should be ~0; a nonzero value here
                               would indicate infer()'s renormalization isn't
                               being hit (e.g. a stale checkpoint/code path).
        'Tm02_RMSE', 'Tm02_Bias' : float, RMSE / mean signed error (predicted
                               − target) of Tm02 = √(m₀/m₂) at the final
                               step, computed directly on the unit-area shape
                               spectrum. Tm02 is scale-invariant (scaling
                               E(f) by any common factor scales m₀ and m₂
                               identically, leaving their ratio untouched),
                               so — unlike Hs — it needs no freq_means/
                               denormalisation to be physically meaningful
                               here; reuses utils.compute_bulk_params (same
                               function the 'density' block above uses) and
                               discards its Hs component, which has no
                               physical meaning on a shape spectrum
                               (normalised to unit area, not real energy).
                               A whole-spectrum average across however many
                               wind-sea/swell partitions the sample has —
                               see 'Tm02_RMSE_windsea'/'Tm02_RMSE_swell'
                               below (compute_peak_metrics=True only) for
                               the partition-conditioned breakdown that
                               doesn't blend those two populations together.

    Peak-aware multimodal metrics (target == 'shape', compute_peak_metrics=True only)
    -------------------------------------------------------------------------------
    A whole-spectrum frequency-weighted RMSE dilutes errors confined to a
    narrow peak (1-2 of 47 bins) almost to invisibility — a model that
    blurs two distinct swell/wind-sea peaks into one smoothed, lower hump
    can show a similar or even BETTER 'Shape_RMSE' than a genuinely
    unimodal sample, since most of the trapz-weighted mass sits in the
    (correctly predicted) shoulders/tail. These metrics use
    utils.peak_modality_metrics (scipy.signal.find_peaks under the hood,
    prominence relative to each spectrum's own max) to surface that failure
    mode directly, using the FINAL forecast step only. As of the 2026-08-06
    switch to utils.spectral_partitioning, peak detection here is the
    Portilla et al. (2009) four-criterion significant-peak test (frequency
    cutoff, minimum partition energy fraction, minimum bin width, no
    "sandwiched" ripples) rather than scipy.signal.find_peaks with a
    prominence threshold relative to each spectrum's own max:
        'Peak_Count_True_Mean' : float, mean number of detected peaks in the
                               true spectrum. Compare against
                               'Peak_Count_Pred_Mean' — a model that
                               over-smooths its output will show a
                               noticeably lower predicted count.
        'Peak_Count_Pred_Mean' : float, same, on the model's prediction.
        'Peak_Height_RelError' : float, mean relative error between
                               predicted and true energy AT each true
                               peak's frequency bin, pooled over every
                               peak across every sample — deliberately not
                               gated on the model having its own detected
                               peak there, so a fully blurred secondary
                               peak still counts against this number.
                               Pooled across BOTH wind-sea and swell
                               partitions — see the _windsea/_swell
                               variants below to avoid blending those two
                               different predictability regimes together.
        'Peak_Separation_Recall': float, fraction of true peaks that have a
                               model-detected peak within a few bins of
                               them — the direct "does the model actually
                               separate the peaks" number; 1.0 only if
                               every true peak has a matching predicted one.
                               Also pooled across both partition types.
        'Peak_Height_RelError_windsea', 'Peak_Height_RelError_swell',
        'Peak_Separation_Recall_windsea', 'Peak_Separation_Recall_swell' :
                               float, the same two definitions above,
                               computed separately per partition — each
                               true partition (Portilla-significant peak +
                               its trough-to-trough window) is classified
                               'wind_sea' vs 'swell' via
                               utils.spectral_partitioning.classify_partition
                               (Violante-Carvalho γ* = S_obs(fp)/S_PM(fp)
                               threshold). Wind-sea partitions are broad,
                               energetic, fast-evolving (a magnitude/energy-
                               tracking problem); swell partitions are
                               narrow, slow, persistent (closer to a
                               position/shift problem, e.g. what a
                               Wasserstein loss term specifically targets)
                               — a loss change that only helps one of the
                               two is invisible in the pooled numbers above.
        'Peak_windsea_n', 'Peak_swell_n' : int, number of true partitions
                               pooled into each label's bucket above.
        'Tm02_RMSE_windsea', 'Tm02_Bias_windsea',
        'Tm02_RMSE_swell', 'Tm02_Bias_swell' : float, RMSE / mean signed
                               error of Tm02 = √(m₀/m₂), integrated over
                               EACH true partition's own window rather than
                               the whole spectrum (see 'Tm02_RMSE' above) —
                               a partition whose predicted in-window energy
                               is below energy_frac of that partition's own
                               true in-window energy is excluded (Tm02 is
                               ill-conditioned on near-zero mass; the miss
                               is already reflected in
                               Peak_Separation_Recall).
        'Tm02_windsea_n', 'Tm02_swell_n' : int, number of partitions pooled
                               into each label's Tm02 bucket (<= the
                               corresponding Peak_*_n — see above).
        'Shape_unimodal_samples', 'Shape_multimodal_samples' : int, sample
                               counts (true peak count == 1 vs >= 2).
        'Shape_RMSE_unimodal', 'Shape_SS_unimodal',
        'Shape_RMSE_multimodal', 'Shape_SS_multimodal' : float, the exact
                               same 'Shape_RMSE'/'Shape_SS' computation
                               above, just bucketed by whether the true
                               final-step spectrum is unimodal or
                               multimodal — lets 'is the model worse on
                               multimodal seas' be read directly off the
                               metrics instead of inferred from plots.

    Note on frequency weighting
    ----------------------------
    The frequency grid is log-spaced (dense near 0.02 Hz, coarse near
    0.485 Hz). 'SI_per_bin' is a per-bin statistic (no aggregation across
    frequency), so bin width is irrelevant there. Every other metric that
    collapses the frequency axis for a 'density'/'shape' target — 'RMSE',
    'CC', 'Bias', 'R2', 'overall_SS', the per_step_* variants, 'Shape_RMSE',
    and 'SI_mean' — uses utils.trapz_weights(freqs) instead of a plain
    arithmetic mean over bins, so the dense low-frequency region isn't
    over-represented relative to its actual share of the physical spectrum.
    This is the same weighting the training loss uses (nn/training_loop.py),
    so validation-time early stopping / LR scheduling / Optuna's 'RMSE'
    objective are consistent with what the model is actually optimizing.
    'hs' has no frequency axis (output_dim=1) and is unaffected throughout.

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

    # Frequency-weighted (trapezoidal) evaluation, matching the training loss
    # (nn/training_loop.py) and Shape_RMSE/SI_mean below. 'hs' has no
    # frequency axis (output_dim=1) so it's left unweighted.
    freq_weights = None
    if model.target != 'hs' and freqs is not None:
        freq_weights = torch.from_numpy(
            trapz_weights(freqs.cpu().numpy())
        ).to(dtype=torch.float32)

    def _weighted_bias(pred, true):
        diff = pred - true
        val = diff.mean() if freq_weights is None else (diff * freq_weights).sum(dim=-1).mean()
        return val.item()

    def _weighted_r2(pred, true):
        if freq_weights is None:
            ss_res = ((true - pred) ** 2).sum()
            ss_tot = ((true - true.mean()) ** 2).sum()
        else:
            true_wmean = (true * freq_weights).sum(dim=-1).mean()
            ss_res = (((true - pred) ** 2) * freq_weights).sum()
            ss_tot = (((true - true_wmean) ** 2) * freq_weights).sum()
        return (1.0 - ss_res / ss_tot).item() if ss_tot > 0 else float('nan')

    def _weighted_cc(pred, true):
        if freq_weights is None:
            return pearson_corrcoef(pred.flatten(), true.flatten()).item()
        p_wmean = (pred * freq_weights).sum(dim=-1).mean()
        t_wmean = (true * freq_weights).sum(dim=-1).mean()
        pc, tc = pred - p_wmean, true - t_wmean
        cov   = ((pc * tc) * freq_weights).sum(dim=-1).mean()
        var_p = ((pc ** 2) * freq_weights).sum(dim=-1).mean()
        var_t = ((tc ** 2) * freq_weights).sum(dim=-1).mean()
        return (cov / torch.sqrt(var_p * var_t)).item()

    with torch.no_grad():
        for src, aux, y_batch in tqdm(dataloader):
            src = src.to(device)
            aux = aux.to(device)
            y_batch = y_batch.to(device)

            # Convert to log-spectral-energy space, matching train_one_epoch
            # — see the NOTE in this function's docstring.
            if model.target == 'density':
                if freq_means is None:
                    raise ValueError("freq_means is required for target='density'")
                fm = freq_means.to(device)
                y_batch = to_log_space(y_batch * fm, fm)
            elif model.target == 'shape':
                if shape_means is None:
                    raise ValueError("shape_means is required for target='shape'")
                y_batch = to_log_space(y_batch, shape_means.to(device))

            # Persistence forecast: last observed value broadcast over all steps
            # get_start_token returns shape (batch, 1) for hs or (batch, num_freqs) for
            # density/shape. freq_means must be forwarded here (not just to model.infer()
            # below) so the hs/shape persistence baseline is denormalised the same way the
            # model's own predictions are — otherwise the SS denominator is wrong.
            start_token = get_start_token(src, model.target, freqs, device,
                                          freq_means=freq_means, shape_means=shape_means)
            # shape → (batch, 1, output_dim) → (batch, lead_time, output_dim)
            persistence = start_token.unsqueeze(1).expand(-1, y_batch.shape[1], -1)

            # Autoregressive model inference — no ground truth in decoder
            y_pred = model.infer(src, freqs, lead_time, freq_means=freq_means,
                                 shape_means=shape_means, aux=aux)

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
        # Not flattened away: keep the trailing frequency axis intact so
        # freq_weights can broadcast against it inside rmse_fn/_weighted_*.
        pred_s = y_pred_all[:, step, :]
        true_s = y_true_all[:, step, :]
        pers_s = y_pers_all[:, step, :]

        rmse_s = rmse_fn(pred_s, true_s, weights=freq_weights).item()
        rmse_p = rmse_fn(pers_s, true_s, weights=freq_weights).item()
        ss_s   = 1.0 - rmse_s / rmse_p if rmse_p > 0 else float('nan')
        bias_s = _weighted_bias(pred_s, true_s)
        r2_s   = _weighted_r2(pred_s, true_s)

        per_step_rmse.append(rmse_s)
        per_step_rmse_pers.append(rmse_p)
        per_step_ss.append(ss_s)
        per_step_bias.append(bias_s)
        per_step_r2.append(r2_s)

    # Overall metrics — kept as (total_samples, lead_time, output_dim) rather
    # than flattened, so freq_weights broadcasts against the trailing
    # frequency axis inside rmse_fn/_weighted_*.
    rmse       = rmse_fn(y_pred_all, y_true_all, weights=freq_weights).item()
    rmse_pers  = rmse_fn(y_pers_all, y_true_all, weights=freq_weights).item()
    cc         = _weighted_cc(y_pred_all, y_true_all)
    overall_ss = 1.0 - rmse / rmse_pers if rmse_pers > 0 else float('nan')
    # For hs target y_pred/y_true are physical Hs, so this is the final-step
    # (i.e. the actual forecast lead time, not an average over the
    # autoregressive steps leading up to it) Skill Score on physical Hs.
    # For density target this is overwritten below using denormalised Hs.
    hs_ss = per_step_ss[-1]
    bias       = _weighted_bias(y_pred_all, y_true_all)
    r2         = _weighted_r2(y_pred_all, y_true_all)

    # Hs_MAPE: for target == 'hs', y_pred/y_true ARE physical Hs already
    # (prepare_y computes compute_hs on the raw, non-normalised density
    # before any scaling is applied). For target == 'density' this is
    # overwritten below once Hs is derived from the denormalised spectrum.
    hs_mape = float(100.0 * torch.mean(
        torch.abs(y_pred_all - y_true_all) / torch.abs(y_true_all)
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
        freq_w   = trapz_weights(freqs_np)                    # (num_freqs,), sums to 1

        # Recover physical E(f): y_pred_all/y_true_all are log-spectral-energy
        # (see docstring NOTE); shape (batch, lead_time, num_freqs)
        pred_np = np.exp(y_pred_all.numpy())
        true_np = np.exp(y_true_all.numpy())

        # Restrict every bulk/shape/SI metric below to the FINAL forecast step
        # only (the actual chosen lead time) rather than pooling across all
        # autoregressive steps — intermediate steps are scaffolding to reach
        # that step, not a deliverable evaluated in their own right (same
        # reasoning as 'final_step_SS' in nn/optimization.py::_compute_val_score
        # and the 'hs'-target Hs_SS above). The lead_time axis is kept at
        # size 1 (not squeezed) so every shape/indexing assumption below
        # (axis=2 for freq, reshape(-1, num_freqs), etc.) still holds.
        pred_np = pred_np[:, -1:, :]
        true_np = true_np[:, -1:, :]

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

        # Hs Skill Score: normalise model RMSE against persistence Hs RMSE so
        # the metric is robust to seq_len variation across Optuna trials.
        pers_np      = np.exp(y_pers_all.numpy())[:, -1:, :]
        hs_pers, _   = compute_bulk_params(pers_np, freqs_np)
        hs_rmse_pers = float(np.sqrt(np.mean((hs_pers - hs_true) ** 2)))
        hs_rmse_model = float(np.sqrt(np.mean(hs_err ** 2)))
        hs_ss = 1.0 - hs_rmse_model / hs_rmse_pers if hs_rmse_pers > 0 else float('nan')

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

            # Frequency-weighted (trapezoidal) mean squared error across bins,
            # not a plain arithmetic mean — see "Note on frequency weighting".
            per_spectrum_rmse = np.sqrt(
                (((shape_pred - shape_true) ** 2) * freq_w).sum(axis=2)   # (batch, lead_time)
            )
            shape_rmse = float(per_spectrum_rmse[valid].mean())

            # Persistence baseline in the same decoupled-from-magnitude shape
            # space, for a Shape Skill Score comparable across target types.
            shape_pers = pers_np / m0_denom
            per_spectrum_rmse_pers = np.sqrt(
                (((shape_pers - shape_true) ** 2) * freq_w).sum(axis=2)
            )
            shape_rmse_pers = float(per_spectrum_rmse_pers[valid].mean())
            shape_ss = (1.0 - shape_rmse / shape_rmse_pers
                        if shape_rmse_pers > 0 else float('nan'))

            # 1-D Wasserstein-2 (quadratic earth-mover) distance, same
            # masked-mean discipline as Shape_RMSE/SS above (near-zero-mass
            # samples excluded before averaging) — see
            # utils.SpectralWassersteinLoss. Reuses the SAME class the
            # 'shape' target uses unchanged: it already internally exp()s +
            # mass-normalizes, so it isn't actually shape-specific (see that
            # class's docstring). Uses y_pred_all[:, -1:, :] (colon-slice,
            # keeping the lead_time axis at size 1) rather than the 'shape'
            # block's integer-index convention, so the result's shape
            # matches `valid`'s (batch, 1) without any reshape.
            wasserstein_fn = SpectralWassersteinLoss()
            w2_per_sample = wasserstein_fn(
                y_pred_all[:, -1:, :], y_true_all[:, -1:, :], freqs, reduction='none'
            ).numpy()  # (batch, 1)
            shape_wasserstein = float(w2_per_sample[valid].mean())
        else:
            shape_rmse = float('nan')
            shape_ss = float('nan')
            shape_wasserstein = float('nan')

        # --- Per-frequency-bin Scatter Index ---
        # Flatten to (N, num_freqs) so each bin's RMSE and mean are computed
        # over all samples at the final forecast step (pred_np/true_np were
        # already sliced to the last step above).  SI[i] = RMSE_i / mean(E_true_i).
        flat_pred = pred_np.reshape(-1, pred_np.shape[2])   # (N, num_freqs)
        flat_true = true_np.reshape(-1, true_np.shape[2])

        rmse_per_bin = np.sqrt(((flat_pred - flat_true) ** 2).mean(axis=0))  # (num_freqs,)
        mean_per_bin = flat_true.mean(axis=0).clip(min=1e-12)                # (num_freqs,)
        si_per_bin   = rmse_per_bin / mean_per_bin                           # (num_freqs,)
        si_mean      = float((si_per_bin * freq_w).sum())   # frequency-weighted, not flat mean

        bulk = {
            'Hs_RMSE'             : hs_rmse_model,
            'Hs_SS'               : hs_ss,
            'Hs_Bias'             : float(np.mean(hs_err)),
            'Hs_MAPE'             : hs_mape,
            'Tm02_RMSE'           : float(np.sqrt(np.mean(tm02_err ** 2))),
            'Tm02_Bias'           : float(np.mean(tm02_err)),
            'Shape_RMSE'          : shape_rmse,
            'Shape_masked_samples': n_masked,
            'Shape_SS'            : shape_ss,
            'Shape_Wasserstein'   : shape_wasserstein,
            'SI_per_bin'          : si_per_bin.tolist(),
            'SI_mean'             : si_mean,
        }

    if model.target == 'shape' and freqs is not None:
        freqs_np = freqs.cpu().numpy()

        # y_pred_all/y_true_all/y_pers_all are log-shape (see docstring
        # NOTE) — exp() back to linear unit-area shape before any physical
        # computation. Shape_RMSE/Shape_SS can no longer alias
        # per_step_rmse[-1]/per_step_ss[-1] (those are now log-space); they
        # must be recomputed here in physical space to stay comparable to
        # the 'density' target's (also physical) Shape_RMSE/Shape_SS above.
        pred_phys_all = torch.exp(y_pred_all)
        true_phys_all = torch.exp(y_true_all)
        pers_phys_all = torch.exp(y_pers_all)

        # Physical-space per-step curve, same loop shape as the top-level
        # (log-space) per_step_* block above — see the '_phys' key docstring
        # entries: this is what makes a 'shape'-target lead-time curve
        # comparable to an always-physical source (linear_baseline, a
        # 'density' target's exp()'d Shape_RMSE, etc.), which the top-level
        # per_step_* keys deliberately are not.
        per_step_rmse_phys, per_step_rmse_pers_phys = [], []
        per_step_ss_phys, per_step_bias_phys = [], []
        for step in range(pred_phys_all.shape[1]):
            p_s, t_s, pr_s = pred_phys_all[:, step, :], true_phys_all[:, step, :], pers_phys_all[:, step, :]
            r_s  = rmse_fn(p_s, t_s, weights=freq_weights).item()
            rp_s = rmse_fn(pr_s, t_s, weights=freq_weights).item()
            per_step_rmse_phys.append(r_s)
            per_step_rmse_pers_phys.append(rp_s)
            per_step_ss_phys.append(1.0 - r_s / rp_s if rp_s > 0 else float('nan'))
            per_step_bias_phys.append(_weighted_bias(p_s, t_s))

        pred_final, true_final, pers_final = pred_phys_all[:, -1, :], true_phys_all[:, -1, :], pers_phys_all[:, -1, :]
        shape_rmse_model = per_step_rmse_phys[-1]
        shape_rmse_pers  = per_step_rmse_pers_phys[-1]
        shape_ss = per_step_ss_phys[-1]

        # --- Tm02 (whole spectrum) ---
        # Tm02 = sqrt(m0/m2) is scale-invariant: scaling E(f) by any common
        # factor scales m0 and m2 identically, leaving their ratio (and thus
        # Tm02) untouched. Unlike Hs, it needs no physical units/freq_means
        # to be meaningful, so it can be computed directly on the unit-area
        # SHAPE spectrum — reuses compute_bulk_params (same function the
        # 'density' target's block above uses) and discards its Hs
        # component, which has no physical meaning on a shape spectrum
        # (normalised to unit area, not real energy).
        _, tm02_pred = compute_bulk_params(pred_final.numpy(), freqs_np)
        _, tm02_true = compute_bulk_params(true_final.numpy(), freqs_np)
        tm02_err = tm02_pred - tm02_true

        # 1-D Wasserstein-2 (quadratic earth-mover) distance between
        # predicted and true shape, via SpectralWassersteinLoss's
        # quantile-domain formula — cheap pure tensor math (no scipy/Python
        # loop, unlike the peak metrics below), so always computed, no
        # opt-in flag needed. Takes the
        # LOG-shape tensors directly (exponentiates internally) — NOT
        # pred_final/true_final above, which are already physical.
        #
        # Computed at every step (not just final) since it answers a
        # different question than per_step_RMSE_phys: RMSE punishes a
        # shifted peak almost as harshly as a vanished one, so a model that
        # tracks a swell peak's position well but blurs its exact height can
        # look no better than one that misses it entirely — per_step_
        # Wasserstein stays low in the former case, high in the latter (see
        # SpectralWassersteinLoss's docstring), so comparing the two curves
        # side by side is the point.
        wasserstein_fn = SpectralWassersteinLoss()
        per_step_wasserstein, per_step_wasserstein_pers = [], []
        for step in range(pred_phys_all.shape[1]):
            per_step_wasserstein.append(
                wasserstein_fn(y_pred_all[:, step, :], y_true_all[:, step, :], freqs).item())
            per_step_wasserstein_pers.append(
                wasserstein_fn(y_pers_all[:, step, :], y_true_all[:, step, :], freqs).item())
        shape_wasserstein = per_step_wasserstein[-1]

        # Per-frequency-bin breakdown of the FINAL step's Wasserstein
        # distance (see reduction='per_bin''s docstring) — mean over the
        # batch axis of the squared quantile gap AT each true bin's own
        # frequency. NOT integrable against `freqs` to recover
        # Shape_Wasserstein (unlike the pre-2026-08-17 W1 version of this
        # comment) — per_bin's domain is cumulative probability (cdf_true),
        # non-uniformly spaced in frequency, so trapz(this, freqs) is not a
        # meaningful quantity; see SpectralWassersteinLoss's docstring.
        # Still lets a peak-shift error be localized along the frequency
        # axis the same way RMSE_per_bin localizes a pointwise-value error,
        # without conflating the two failure modes.
        wasserstein_per_bin = wasserstein_fn(
            y_pred_all[:, -1, :], y_true_all[:, -1, :], freqs, reduction='per_bin'
        ).mean(dim=0).numpy()
        wasserstein_per_bin_pers = wasserstein_fn(
            y_pers_all[:, -1, :], y_true_all[:, -1, :], freqs, reduction='per_bin'
        ).mean(dim=0).numpy()

        # Mass-conservation diagnostic: model.infer() already renormalizes
        # each predicted step to unit area (in linear space, before
        # converting back to log-space for the next decode step — see
        # nn/transformer.py::infer()), so this should be ~0 — see the
        # 'Shape_Mass_Error' docstring entry above. Deliberately NOT restricted
        # to the final step (unlike Shape_RMSE/Shape_SS above) — it's a
        # sanity check for a renormalization regression at ANY step, not a
        # lead-time performance metric.
        mass = np.trapezoid(torch.exp(y_pred_all).numpy(), freqs_np, axis=2)  # (batch, lead_time)

        # --- Per-frequency-bin metrics (FINAL forecast step only), physical
        # space --- mirrors the 'density' target's SI_per_bin block above
        # (same SI[i] = RMSE_i / mean(E_true_i) formula), computed on the
        # exp()'d unit-area shape rather than a physical density spectrum —
        # no M0_MASK_THRESHOLD masking needed here for the same reason the
        # rest of this block skips it (prepare_y discards m₀ for 'shape').
        # RMSE_per_bin_pers is included alongside so a per-bin plot can draw
        # the persistence baseline as a reference curve, same as per_step_*.
        pred_np_final = pred_final.numpy()
        true_np_final = true_final.numpy()
        pers_np_final = pers_final.numpy()
        rmse_per_bin      = np.sqrt(((pred_np_final - true_np_final) ** 2).mean(axis=0))  # (num_freqs,)
        rmse_per_bin_pers = np.sqrt(((pers_np_final - true_np_final) ** 2).mean(axis=0))
        bias_per_bin      = (pred_np_final - true_np_final).mean(axis=0)
        mean_per_bin      = true_np_final.mean(axis=0).clip(min=1e-12)
        si_per_bin        = rmse_per_bin / mean_per_bin

        bulk = {
            'per_step_RMSE_phys'     : per_step_rmse_phys,
            'per_step_RMSE_pers_phys': per_step_rmse_pers_phys,
            'per_step_SS_phys'       : per_step_ss_phys,
            'per_step_Bias_phys'     : per_step_bias_phys,
            'RMSE_per_bin'           : rmse_per_bin.tolist(),
            'RMSE_per_bin_pers'      : rmse_per_bin_pers.tolist(),
            'Bias_per_bin'           : bias_per_bin.tolist(),
            'SI_per_bin'             : si_per_bin.tolist(),
            'per_step_Wasserstein'     : per_step_wasserstein,
            'per_step_Wasserstein_pers': per_step_wasserstein_pers,
            'Wasserstein_per_bin'      : wasserstein_per_bin.tolist(),
            'Wasserstein_per_bin_pers' : wasserstein_per_bin_pers.tolist(),
            'Shape_RMSE'       : shape_rmse_model,
            'Shape_SS'         : shape_ss,
            'Shape_Wasserstein': shape_wasserstein,
            'Shape_Mass_Error' : float(np.mean(np.abs(mass - 1.0))),
            'Tm02_RMSE'        : float(np.sqrt(np.mean(tm02_err ** 2))),
            'Tm02_Bias'        : float(np.mean(tm02_err)),
        }

        if compute_peak_metrics:
            peak_metrics, multimodal_mask = peak_modality_metrics(
                freqs_np, pred_final.numpy(), true_final.numpy())
            mask_t = torch.from_numpy(multimodal_mask)

            def _bucket(mask):
                if mask.sum() == 0:
                    return float('nan'), float('nan')
                r  = rmse_fn(pred_final[mask], true_final[mask], weights=freq_weights).item()
                rp = rmse_fn(pers_final[mask], true_final[mask], weights=freq_weights).item()
                return r, (1.0 - r / rp if rp > 0 else float('nan'))

            rmse_uni, ss_uni     = _bucket(~mask_t)
            rmse_multi, ss_multi = _bucket(mask_t)
            bulk.update({
                **peak_metrics,
                'Shape_unimodal_samples'  : int((~multimodal_mask).sum()),
                'Shape_multimodal_samples': int(multimodal_mask.sum()),
                'Shape_RMSE_unimodal'     : rmse_uni,
                'Shape_SS_unimodal'       : ss_uni,
                'Shape_RMSE_multimodal'   : rmse_multi,
                'Shape_SS_multimodal'     : ss_multi,
            })

    metrics = {
        'RMSE'               : rmse,
        'Hs_MAPE'            : hs_mape,
        'Hs_SS'              : hs_ss,
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

    if return_arrays:
        return metrics, (y_pred_all, y_true_all, y_pers_all)
    return metrics
