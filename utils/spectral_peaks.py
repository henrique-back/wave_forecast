import numpy as np

from .spectral_partitioning import find_significant_peaks


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

    Both spectra inputs must already be physical / linear (i.e. exp()'d out
    of log-space by the caller) and on the SAME frequency grid `freqs` (both
    for peak detection — see find_spectral_peaks — and for peaks being
    compared by aligned bin index, not by Hz).

    Parameters
    ----------
    freqs : np.ndarray, shape (num_freqs,) — frequency grid [Hz], forwarded
        to find_spectral_peaks for both pred and true spectra.
    pred_final, true_final : np.ndarray, shape (N, num_freqs)
    f_max, energy_frac, min_bins : forwarded to find_spectral_peaks (the
        Portilla et al. 2009 significant-peak criteria) for both pred and
        true spectra.
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
            sits", so a fully blurred secondary peak still counts).
        'Peak_Separation_Recall' — fraction of true peaks with a predicted
            peak within bin_tolerance bins; the direct "does the model
            separate the peaks" number.
    multimodal_mask : np.ndarray[bool], shape (N,) — true peak count >= 2.
        Returned separately (not a metrics-dict entry: not JSON-safe, and
        callers already have their own RMSE/frequency-weighting in scope)
        so a caller can bucket ITS OWN Shape_RMSE/SS by this mask without
        this module needing to know about RMSE weighting conventions.
    """
    pred_final = np.asarray(pred_final)
    true_final = np.asarray(true_final)
    n = true_final.shape[0]

    true_counts = np.zeros(n, dtype=int)
    pred_counts = np.zeros(n, dtype=int)
    rel_errors = []
    n_true_peaks_total = 0
    n_recalled = 0

    for i in range(n):
        true_peaks = find_spectral_peaks(freqs, true_final[i], f_max, energy_frac, min_bins)
        pred_peaks = find_spectral_peaks(freqs, pred_final[i], f_max, energy_frac, min_bins)
        true_counts[i] = len(true_peaks)
        pred_counts[i] = len(pred_peaks)

        for b in true_peaks:
            true_h = max(true_final[i, b], 1e-8)
            rel_errors.append(abs(pred_final[i, b] - true_final[i, b]) / true_h)
            n_true_peaks_total += 1
            if pred_peaks.size > 0 and np.min(np.abs(pred_peaks - b)) <= bin_tolerance:
                n_recalled += 1

    metrics = {
        'Peak_Count_True_Mean': float(true_counts.mean()) if n > 0 else float('nan'),
        'Peak_Count_Pred_Mean': float(pred_counts.mean()) if n > 0 else float('nan'),
        'Peak_Height_RelError': float(np.mean(rel_errors)) if rel_errors else float('nan'),
        'Peak_Separation_Recall': (n_recalled / n_true_peaks_total
                                    if n_true_peaks_total > 0 else float('nan')),
    }
    multimodal_mask = true_counts >= 2
    return metrics, multimodal_mask
