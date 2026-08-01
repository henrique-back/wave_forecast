import numpy as np
from scipy.signal import find_peaks


def find_spectral_peaks(spectrum, prominence_frac=0.25):
    """
    Detect local-maxima bin indices in a single 1-D spectrum.

    prominence_frac is relative to THIS spectrum's own max, not a global
    constant — makes the detector scale-free across samples of very
    different energy (works the same on a unit-area 'shape' spectrum or a
    physical 'density' spectrum spanning orders of magnitude in m0).

    0.25 was chosen empirically (not the initial 0.08 guess) by sweeping
    prominence_frac over shape_v11's full lead_12h test set: 0.08 counted
    an implausible mean of ~2.9 "peaks" per true spectrum (85% of samples
    flagged multimodal) — clearly picking up minor ripples, not distinct
    wave systems. 0.25 gives a mean of ~1.6 peaks (49% multimodal, a much
    more balanced and physically plausible split) while still resolving
    exactly the 2 real peaks on a known visually-bimodal sample
    (results/shape_v11/shape/lead_12h/inference_plots/sample_2572.png)
    consistently across prominence_frac in [0.15, 0.30] — i.e. 0.25 sits
    in the middle of a stable range, not right at a knife-edge threshold.

    Note: scipy.signal.find_peaks cannot flag a peak at index 0 or -1 (no
    two-sided neighbor to compare against). Per CLAUDE.md the buoy's
    0.02-0.485 Hz grid has density ~0 at both ends for typical sea states,
    so this is a rare edge case — documented here rather than engineered
    around.

    Parameters
    ----------
    spectrum : np.ndarray, shape (num_freqs,)
    prominence_frac : float

    Returns
    -------
    np.ndarray[int] — peak bin indices (possibly empty).
    """
    spectrum = np.asarray(spectrum)
    if spectrum.size == 0 or spectrum.max() <= 0:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(spectrum, prominence=spectrum.max() * prominence_frac)
    return peaks


def peak_modality_metrics(pred_final, true_final, prominence_frac=0.25, bin_tolerance=2):
    """
    Peak-fidelity metrics for a batch of final-forecast-step spectra, aimed
    at surfacing the multimodal (double/triple-peaked) sea-state failure
    mode that a whole-spectrum frequency-weighted RMSE dilutes away — a
    peak confined to 1-2 of 47 bins barely moves an integral over the
    whole grid, even when the model badly misses its height or blurs two
    peaks into one smoothed hump.

    Both inputs must already be physical / linear (i.e. exp()'d out of
    log-space by the caller) and on the same frequency grid (peaks are
    compared by aligned bin index, not by Hz — no `freqs` argument needed).

    Parameters
    ----------
    pred_final, true_final : np.ndarray, shape (N, num_freqs)
    prominence_frac : float — forwarded to find_spectral_peaks for both
        pred and true spectra.
    bin_tolerance : int — a true peak counts as "separated" by the model
        if any predicted peak (detected with the SAME prominence_frac) lies
        within this many bins of it — i.e. the model's own curve must show
        a locally-prominent max near the true peak, not just a non-zero
        value there.

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
        true_peaks = find_spectral_peaks(true_final[i], prominence_frac)
        pred_peaks = find_spectral_peaks(pred_final[i], prominence_frac)
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
