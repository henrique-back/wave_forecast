"""
Per-sample sufficient statistics for peak fidelity, so the score can be
re-pooled over any subset of samples (a rolling time window, a regime, a
bootstrap resample) or of peaks (a frequency band) without re-running peak
detection.

peak_fidelity (nn/optimization.py::_compute_val_score) is a pooled score —
F1(pooled precision, macro recall) x (1 - min(macro rel_err, 1)) — so it is
not defined per sample: with 1-3 peaks per spectrum a single sample's score
is almost always 0 or 1. What IS additive per sample are its ingredients:
predicted-peak and matched counts, and per partition label the true-peak
count, the recalled count and the summed relative height error. peak_stats
returns those, and pooled_peak_fidelity re-pools them with sample weights
(0/1 for a subset, multiplicities for a bootstrap draw).

Detection, matching and labelling follow utils/spectral_peaks.py::
peak_modality_metrics exactly (that function is left untouched so running
jobs keep importing identical code); tests/test_peak_records.py pins the
parity of every pooled ingredient and of the final score.

Frequency-band groups: a true peak is assigned to the band holding its own
fp, and so is a predicted peak. Unlike the wind-sea/swell label, which a
predicted peak matching nothing cannot inherit (decision 031), a band is
defined for every peak, so the full score (precision included) is defined
per band. A true peak and its matching predicted peak can straddle a band
edge; each is then counted in its own band.
"""
import numpy as np

from .spectral_partitioning import find_peak_windows, classify_partition
from .spectral_peaks import find_spectral_peaks

LABELS = ('wind_sea', 'swell')

# Column layout of the last axis of peak_stats' output.
N_PRED, N_PRED_MATCHED = 0, 1
N_TRUE = {'wind_sea': 2, 'swell': 5}
N_RECALLED = {'wind_sea': 3, 'swell': 6}
REL_ERR_SUM = {'wind_sea': 4, 'swell': 7}
N_STATS = 8


def peak_stats(freqs, pred, true, band_edges=(), f_max=0.4, energy_frac=0.05,
               min_bins=2, bin_tolerance=2):
    """
    Parameters
    ----------
    freqs : np.ndarray (F,)
    pred, true : np.ndarray (N, F) — physical spectra (m^2/Hz: the labels use
        gamma* against Pierson-Moskowitz, see decision 029) on `freqs`.
    band_edges : sequence of float — interior band edges in Hz. Bands are
        [edges[i-1], edges[i]) with -inf/+inf at the ends.
    f_max, energy_frac, min_bins, bin_tolerance : as peak_modality_metrics.

    Returns
    -------
    stats : np.ndarray (N, 1 + len(band_edges) + 1, N_STATS) — group 0 is
        every peak, group 1 + b is band b.
    n_true_peaks : np.ndarray (N,) int — true significant peak count.
    dominant_is_swell : np.ndarray (N,) bool — label of the highest true
        peak (False also when the true spectrum has no peak).
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    edges = np.asarray(band_edges, dtype=np.float64)
    n = true.shape[0]
    stats = np.zeros((n, len(edges) + 2, N_STATS))
    n_true_peaks = np.zeros(n, dtype=int)
    dominant_is_swell = np.zeros(n, dtype=bool)

    def groups(fp):
        return (0, 1 + int(np.searchsorted(edges, fp, side='right')))

    for i in range(n):
        true_spec, pred_spec = true[i], pred[i]
        true_windows = find_peak_windows(freqs, true_spec, f_max, energy_frac, min_bins)
        pred_peaks = find_spectral_peaks(freqs, pred_spec, f_max, energy_frac, min_bins)
        true_idxs = np.array([w[0] for w in true_windows], dtype=int)
        n_true_peaks[i] = len(true_idxs)

        for p in pred_peaks:
            matched = true_idxs.size > 0 and np.abs(true_idxs - p).min() <= bin_tolerance
            for g in groups(freqs[p]):
                stats[i, g, N_PRED] += 1
                stats[i, g, N_PRED_MATCHED] += matched

        best_h = -np.inf
        for peak_idx, _, _ in true_windows:
            rel_err = abs(pred_spec[peak_idx] - true_spec[peak_idx]) / max(true_spec[peak_idx], 1e-8)
            recalled = pred_peaks.size > 0 and np.abs(pred_peaks - peak_idx).min() <= bin_tolerance
            label = classify_partition(fp=freqs[peak_idx], S_obs_at_fp=true_spec[peak_idx])
            for g in groups(freqs[peak_idx]):
                stats[i, g, N_TRUE[label]] += 1
                stats[i, g, N_RECALLED[label]] += recalled
                stats[i, g, REL_ERR_SUM[label]] += rel_err
            if true_spec[peak_idx] > best_h:
                best_h = true_spec[peak_idx]
                dominant_is_swell[i] = label == 'swell'

    return stats, n_true_peaks, dominant_is_swell


def _ratio(num, den):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def pooled_peak_fidelity(stats, weights=None, label=None):
    """
    Pool peak_stats rows and return the score and its components.

    Parameters
    ----------
    stats : np.ndarray (..., N, N_STATS) — one group's rows; leading axes
        (e.g. forecast step) are kept.
    weights : np.ndarray (N,) or (B, N) | None — per-sample weights: a 0/1
        subset mask, or bootstrap multiplicities (a leading B axis gives B
        scores at once). None weights every sample 1.
    label : None | 'wind_sea' | 'swell'
        None: the peak_fidelity of _compute_val_score —
            F1(precision, macro recall over labels) x (1 - min(macro rel_err, 1)).
        a label: recall_label x (1 - min(rel_err_label, 1)). Precision has no
            per-label form (a predicted peak matching nothing has no true
            partition to take a label from — decision 031), so this is the
            recall-and-height part of the score only, for that label.

    Returns
    -------
    dict of np.ndarray: 'peak_fidelity', 'precision', 'recall', 'rel_err',
    'n_true', 'n_pred'. NaN where the score is undefined (no true peak in
    the pool) — _compute_val_score returns -inf there instead, which suits
    model selection but not plotting.
    """
    stats = np.asarray(stats, dtype=np.float64)
    if weights is None:
        weights = np.ones(stats.shape[-2])
    w = np.asarray(weights, dtype=np.float64)
    # Sum over the sample axis: (B?, N) x (..., N, S) -> (B?, ..., S)
    tot = np.tensordot(w, stats, axes=([w.ndim - 1], [stats.ndim - 2]))

    n_pred = tot[..., N_PRED]
    precision = _ratio(tot[..., N_PRED_MATCHED], n_pred)
    recall_l = {l: _ratio(tot[..., N_RECALLED[l]], tot[..., N_TRUE[l]]) for l in LABELS}
    rel_err_l = {l: _ratio(tot[..., REL_ERR_SUM[l]], tot[..., N_TRUE[l]]) for l in LABELS}

    if label is None:
        with np.errstate(invalid='ignore'):
            recall = _nanmean2(recall_l['wind_sea'], recall_l['swell'])
            rel_err = _nanmean2(rel_err_l['wind_sea'], rel_err_l['swell'])
        n_true = tot[..., N_TRUE['wind_sea']] + tot[..., N_TRUE['swell']]
        p = np.where(np.isnan(precision), 0.0, precision)   # no prediction at all scores 0
        with np.errstate(invalid='ignore', divide='ignore'):
            f1 = np.where(p + recall > 0, 2 * p * recall / np.where(p + recall > 0, p + recall, 1.0), 0.0)
        score = f1 * (1.0 - np.minimum(rel_err, 1.0))
    else:
        recall, rel_err = recall_l[label], rel_err_l[label]
        n_true = tot[..., N_TRUE[label]]
        score = recall * (1.0 - np.minimum(rel_err, 1.0))
    score = np.where(np.isnan(recall) | np.isnan(rel_err), np.nan, score)
    return {'peak_fidelity': score, 'precision': precision, 'recall': recall,
            'rel_err': rel_err, 'n_true': n_true, 'n_pred': n_pred}


def _nanmean2(a, b):
    """Elementwise nanmean of two arrays, NaN (no warning) where both are NaN."""
    both = np.isnan(a) & np.isnan(b)
    s = np.where(np.isnan(a), 0.0, a) + np.where(np.isnan(b), 0.0, b)
    c = (~np.isnan(a)).astype(float) + (~np.isnan(b)).astype(float)
    return np.where(both, np.nan, s / np.where(c > 0, c, 1.0))
