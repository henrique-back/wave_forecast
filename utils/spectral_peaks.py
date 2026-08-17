import numpy as np

from .spectral_partitioning import find_significant_peaks, find_peak_windows, classify_partition


def find_spectral_peaks(freqs, spectrum, f_max=0.4, energy_frac=0.05, min_bins=2):
    """
    Detect physically-significant peak bin indices in a single 1-D spectrum.

    Thin wrapper around utils.spectral_partitioning.find_significant_peaks —
    the Portilla et al. (2009, section 2b.2) four-criterion spurious-peak
    test:
      1. fp > f_max (0.35-0.4 Hz) — reject high-frequency tail noise.
      2. partition energy < energy_frac * E_total — reject peaks that carry
         too small a share of the spectrum's total energy to be a real,
         separate wave system.
      3. fewer than min_bins spectral bins on either side of the peak before
         the next trough — reject peaks too narrow to be resolved by the
         grid.
      4. the peak sits between two higher-energy neighboring peaks (a local
         "sandwich") — reject a minor ripple riding on the shoulder of a
         bigger partition.

    This replaces the earlier scale-free prominence_frac=0.25 heuristic
    (chosen by sweeping prominence_frac over shape_v11's lead_12h test set
    until the mean peak count "looked" physically plausible — see git
    history) with published, physically-motivated criteria that don't
    depend on tuning a constant against one buoy's visual impression of how
    many peaks are real.

    Note: as with the old prominence-based detector, scipy.signal.find_peaks
    (used internally by find_significant_peaks) cannot flag a peak at index
    0 or -1 (no two-sided neighbor to compare against). Per CLAUDE.md the
    buoy's 0.02-0.485 Hz grid has density ~0 at both ends for typical sea
    states, so this is a rare edge case — documented here rather than
    engineered around.

    Parameters
    ----------
    freqs    : np.ndarray, shape (num_freqs,) — frequency grid [Hz].
               Required (unlike the old prominence-only detector) because
               criteria 1 and 2 are physical, not bin-index, quantities.
    spectrum : np.ndarray, shape (num_freqs,)
    f_max, energy_frac, min_bins : forwarded to
        utils.spectral_partitioning.find_significant_peaks — see its
        docstring for the full criteria definitions.

    Returns
    -------
    np.ndarray[int] — peak bin indices (possibly empty).
    """
    freqs = np.asarray(freqs)
    spectrum = np.asarray(spectrum)
    if spectrum.size == 0 or spectrum.max() <= 0:
        return np.array([], dtype=int)
    peaks = find_significant_peaks(
        freqs, spectrum, f_max=f_max, energy_frac=energy_frac, min_bins=min_bins
    )
    return np.array(peaks, dtype=int)


def peak_modality_metrics(freqs, pred_final, true_final, f_max=0.4, energy_frac=0.05,
                           min_bins=2, bin_tolerance=2):
    """
    Peak-fidelity metrics for a batch of final-forecast-step spectra, aimed
    at surfacing the multimodal (double/triple-peaked) sea-state failure
    mode that a whole-spectrum frequency-weighted RMSE dilutes away — a
    peak confined to 1-2 of 47 bins barely moves an integral over the
    whole grid, even when the model badly misses its height or blurs two
    peaks into one smoothed hump.

    Every true partition (a Portilla-significant peak plus its
    trough-to-trough window, from find_peak_windows) is also classified
    'wind_sea' vs 'swell' via classify_partition (Violante-Carvalho
    gamma* = S_obs(fp)/S_PM(fp) threshold), and three per-partition
    quantities — peak height relative error, separation recall, and
    Tm02 = sqrt(m0/m2) period error, all computed WITHIN that partition's
    own window — are pooled both overall (unsuffixed keys, backward
    compatible with the pre-partition-aware version of this function) and
    separately per label (_windsea / _swell suffixes). Pooling across both
    populations hides exactly the attribution an ablation needs: wind-sea
    partitions are broad, energetic, fast-evolving (closer to a
    magnitude/energy-tracking problem), swell partitions are narrow, slow,
    persistent (closer to a position/shift problem, which e.g. a
    Wasserstein loss term specifically targets) — a loss change that only
    helps one of the two would be invisible in the pooled number alone.

    All partition geometry (windows AND labels) comes from the TRUE
    spectrum only, applied to both pred and true. This mirrors
    utils.loss.SoftPeakHeightLoss (tau_k/H_true computed from the true
    window only) and this function's own pre-existing convention for
    Peak_Height_RelError ("evaluated AT each true peak's bin... the
    harshest read"), and it sidesteps partition-matching ambiguity: a
    partition the model misses entirely doesn't need reconciling against
    some predicted partition — it's simply absent from the model's own
    peak list and shows up as a Peak_Separation_Recall miss, not smuggled
    into a height/period error number via some invented predicted window.

    Both spectra inputs must already be physical / linear (i.e. exp()'d out
    of log-space by the caller) and on the SAME frequency grid `freqs` (both
    for peak detection — see find_spectral_peaks/find_peak_windows — and
    for peaks being compared by aligned bin index, not by Hz).

    Parameters
    ----------
    freqs : np.ndarray, shape (num_freqs,) — frequency grid [Hz], forwarded
        to find_peak_windows/find_spectral_peaks for both pred and true
        spectra.
    pred_final, true_final : np.ndarray, shape (N, num_freqs)
    f_max, energy_frac, min_bins : forwarded to find_peak_windows /
        find_spectral_peaks (the Portilla et al. 2009 significant-peak
        criteria) for both true-partition detection and pred-peak
        detection. energy_frac is ALSO reused below as the Tm02 masking
        threshold — a partition whose predicted in-window energy falls
        below energy_frac * that partition's own true in-window energy is
        excluded from the Tm02 buckets (Tm02 is ill-conditioned on
        near-zero mass, and "the model missed this partition" is already
        Peak_Separation_Recall's job, not this metric's) — one "how much
        energy counts as real" constant reused for both jobs rather than a
        second unrelated tunable.
    bin_tolerance : int — a true peak counts as "separated" by the model
        if any predicted peak (detected with the SAME criteria) lies
        within this many bins of it — i.e. the model's own curve must show
        a physically-significant local max near the true peak, not just a
        non-zero value there.

    Returns
    -------
    (metrics, multimodal_mask)
    metrics : dict[str, float] — JSON-safe:
        'Peak_Count_True_Mean', 'Peak_Count_Pred_Mean' — mean detected peak
            count; a model that over-smooths its output will show a lower
            Peak_Count_Pred_Mean than Peak_Count_True_Mean.
        'Peak_Height_RelError' — mean |pred[bin]-true[bin]| / true[bin]
            evaluated AT each true peak's bin, aggregated over every peak
            across every sample (not gated on the model having detected a
            peak there itself — this is deliberately the harshest read:
            "how far off is the model's value at the spot the true peak
            sits", so a fully blurred secondary peak still counts). Pooled
            across BOTH partition labels — kept for backward compatibility;
            prefer the _windsea/_swell variants below when attributing a
            change to a specific failure mode.
        'Peak_Separation_Recall' — fraction of true peaks with a predicted
            peak within bin_tolerance bins; the direct "does the model
            separate the peaks" number. Also pooled across both labels.
        'Peak_Height_RelError_windsea', 'Peak_Height_RelError_swell' :
            float — same definition as 'Peak_Height_RelError', pooled
            within that partition label only.
        'Peak_Separation_Recall_windsea', 'Peak_Separation_Recall_swell' :
            float — same definition as 'Peak_Separation_Recall', pooled
            within that partition label only.
        'Peak_windsea_n', 'Peak_swell_n' : int — number of true partitions
            pooled into each label's height/recall buckets above.
        'Tm02_RMSE_windsea', 'Tm02_Bias_windsea',
        'Tm02_RMSE_swell', 'Tm02_Bias_swell' : float — RMSE / mean signed
            error (pred − true) of Tm02 = sqrt(m0/m2), integrated over EACH
            true partition's own trough-to-trough window (both pred and
            true restricted to that same true-derived window before
            computing their own m0/m2) — scale-invariant (m0, m2 scale
            together under any uniform rescaling of E(f), so their ratio
            doesn't), so meaningful without freq_means/denormalisation even
            on a 'shape' target.
        'Tm02_windsea_n', 'Tm02_swell_n' : int — number of partitions
            pooled into each label's Tm02 buckets (<= the corresponding
            Peak_*_n, since the energy mask above can additionally exclude
            partitions the height/recall metrics still count).
    multimodal_mask : np.ndarray[bool], shape (N,) — true peak count >= 2.
        Returned separately (not a metrics-dict entry: not JSON-safe, and
        callers already have their own RMSE/frequency-weighting in scope)
        so a caller can bucket ITS OWN Shape_RMSE/SS by this mask without
        this module needing to know about RMSE weighting conventions.
    """
    freqs = np.asarray(freqs)
    pred_final = np.asarray(pred_final)
    true_final = np.asarray(true_final)
    n = true_final.shape[0]

    true_counts = np.zeros(n, dtype=int)
    pred_counts = np.zeros(n, dtype=int)

    # Pooled across both labels — backward-compatible top-line numbers.
    rel_errors = []
    n_true_peaks_total = 0
    n_recalled = 0

    # Pooled per label.
    rel_errors_by_label = {'wind_sea': [], 'swell': []}
    recalled_by_label = {'wind_sea': [], 'swell': []}
    tm02_err_by_label = {'wind_sea': [], 'swell': []}

    for i in range(n):
        true_spec = true_final[i]
        pred_spec = pred_final[i]

        # find_peak_windows detects the significant TRUE peaks AND their
        # trough-to-trough windows in one pass, rather than a separate
        # find_spectral_peaks(true) call followed by a separate
        # find_peak_windows(true) call re-running the same
        # scipy.signal.find_peaks + Portilla criteria on the identical
        # spectrum twice.
        true_windows = find_peak_windows(freqs, true_spec, f_max, energy_frac, min_bins)
        pred_peaks = find_spectral_peaks(freqs, pred_spec, f_max, energy_frac, min_bins)

        true_counts[i] = len(true_windows)
        pred_counts[i] = len(pred_peaks)

        for peak_idx, left, right in true_windows:
            true_h = max(true_spec[peak_idx], 1e-8)
            rel_err = abs(pred_spec[peak_idx] - true_spec[peak_idx]) / true_h
            recalled = bool(pred_peaks.size > 0
                             and np.min(np.abs(pred_peaks - peak_idx)) <= bin_tolerance)

            rel_errors.append(rel_err)
            n_true_peaks_total += 1
            if recalled:
                n_recalled += 1

            label = classify_partition(fp=freqs[peak_idx], S_obs_at_fp=true_spec[peak_idx])
            rel_errors_by_label[label].append(rel_err)
            recalled_by_label[label].append(recalled)

            # Tm02 within this true-derived window, both sides integrated
            # over the SAME window (see docstring: geometry always comes
            # from the true side).
            f_win = freqs[left:right + 1]
            true_win = true_spec[left:right + 1]
            pred_win = pred_spec[left:right + 1]

            true_m0 = float(np.trapezoid(true_win, f_win))
            pred_m0 = float(np.trapezoid(pred_win, f_win))
            if pred_m0 >= energy_frac * true_m0:
                true_m2 = max(float(np.trapezoid(true_win * f_win ** 2, f_win)), 1e-12)
                pred_m2 = max(float(np.trapezoid(pred_win * f_win ** 2, f_win)), 1e-12)
                true_tm02 = np.sqrt(max(true_m0, 0.0) / true_m2)
                pred_tm02 = np.sqrt(max(pred_m0, 0.0) / pred_m2)
                tm02_err_by_label[label].append(pred_tm02 - true_tm02)
            # else: the model predicted less than energy_frac of this
            # partition's own energy inside its window — Tm02 is
            # ill-conditioned there, and the miss is already reflected in
            # Peak_Separation_Recall, so this partition is excluded rather
            # than contributing a noisy/undefined period error.

    def _mean_or_nan(vals):
        return float(np.mean(vals)) if vals else float('nan')

    def _rmse_bias(errs):
        if not errs:
            return float('nan'), float('nan')
        arr = np.asarray(errs, dtype=float)
        return float(np.sqrt(np.mean(arr ** 2))), float(np.mean(arr))

    tm02_rmse_windsea, tm02_bias_windsea = _rmse_bias(tm02_err_by_label['wind_sea'])
    tm02_rmse_swell, tm02_bias_swell = _rmse_bias(tm02_err_by_label['swell'])

    metrics = {
        'Peak_Count_True_Mean': float(true_counts.mean()) if n > 0 else float('nan'),
        'Peak_Count_Pred_Mean': float(pred_counts.mean()) if n > 0 else float('nan'),
        'Peak_Height_RelError': float(np.mean(rel_errors)) if rel_errors else float('nan'),
        'Peak_Separation_Recall': (n_recalled / n_true_peaks_total
                                    if n_true_peaks_total > 0 else float('nan')),

        'Peak_Height_RelError_windsea': _mean_or_nan(rel_errors_by_label['wind_sea']),
        'Peak_Height_RelError_swell'  : _mean_or_nan(rel_errors_by_label['swell']),
        'Peak_Separation_Recall_windsea': _mean_or_nan(recalled_by_label['wind_sea']),
        'Peak_Separation_Recall_swell'  : _mean_or_nan(recalled_by_label['swell']),
        'Peak_windsea_n': len(rel_errors_by_label['wind_sea']),
        'Peak_swell_n'  : len(rel_errors_by_label['swell']),

        'Tm02_RMSE_windsea': tm02_rmse_windsea,
        'Tm02_Bias_windsea': tm02_bias_windsea,
        'Tm02_RMSE_swell'  : tm02_rmse_swell,
        'Tm02_Bias_swell'  : tm02_bias_swell,
        'Tm02_windsea_n': len(tm02_err_by_label['wind_sea']),
        'Tm02_swell_n'  : len(tm02_err_by_label['swell']),
    }
    multimodal_mask = true_counts >= 2
    return metrics, multimodal_mask
