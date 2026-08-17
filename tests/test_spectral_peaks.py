"""
Tests for utils/spectral_peaks.py — peak detection and the peak-aware
multimodal metrics used by nn/evaluate.py (target == 'shape',
compute_peak_metrics=True) to surface the failure mode where a
whole-spectrum frequency-weighted RMSE dilutes errors confined to a single
narrow (1-2 bin) peak almost to invisibility.

Reuses the JONSWAP generator and frequency grid from tests/test_spectral.py
(tests/ is a real package — __init__.py present — so this cross-import is
safe and doesn't duplicate pytest collection).
"""

import sys
import os

import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_spectral import _jonswap, FREQS
from utils.spectral_partitioning import classify_partition, find_peak_windows
from utils.spectral_peaks import find_spectral_peaks, peak_modality_metrics


def _bimodal(freqs):
    """Two comparably-sized, well-separated peaks (Tp=7s and Tp=14s)."""
    return _jonswap(freqs, 2.0, 7.0) + _jonswap(freqs, 2.0, 14.0)


class TestFindSpectralPeaks:
    def test_single_jonswap_has_one_peak(self):
        spectrum = _jonswap(FREQS, 2.0, 10.0)
        peaks = find_spectral_peaks(FREQS, spectrum)
        assert len(peaks) == 1

    def test_sum_of_two_separated_jonswaps_has_two_peaks(self):
        spectrum = _bimodal(FREQS)
        peaks = find_spectral_peaks(FREQS, spectrum)
        assert len(peaks) == 2

    def test_zero_spectrum_returns_empty_without_raising(self):
        spectrum = np.zeros_like(FREQS)
        peaks = find_spectral_peaks(FREQS, spectrum)
        assert len(peaks) == 0

    def test_flat_spectrum_returns_empty_without_raising(self):
        spectrum = np.ones_like(FREQS)
        peaks = find_spectral_peaks(FREQS, spectrum)
        assert len(peaks) == 0


class TestPeakModalityMetrics:
    def test_identical_pred_and_true_mixed_batch(self):
        """One unimodal + one bimodal sample, pred == true exactly."""
        unimodal = _jonswap(FREQS, 2.0, 10.0)
        bimodal = _bimodal(FREQS)
        true = np.stack([unimodal, bimodal])
        pred = true.copy()

        metrics, mask = peak_modality_metrics(FREQS, pred, true)

        assert list(mask) == [False, True]
        assert metrics['Peak_Count_True_Mean'] == pytest.approx(1.5)
        assert metrics['Peak_Separation_Recall'] == pytest.approx(1.0)
        assert metrics['Peak_Height_RelError'] == pytest.approx(0.0, abs=1e-6)

    def test_blurred_bimodal_prediction_misses_separation(self):
        """A pred that merges the true spectrum's two peaks into one smoothed
        hump (mimicking the model's real blurring behavior on multimodal
        seas, e.g. results/shape_v11/.../sample_2572.png) must score below
        perfect recall — this is the test that proves the metric actually
        catches that failure, not just that the code runs.
        """
        true = _bimodal(FREQS)
        true_peaks = find_spectral_peaks(FREQS, true)
        assert len(true_peaks) == 2  # precondition for this test to mean anything

        # size=15 uniform smoothing merges the two peaks into a single hump
        # that sits outside bin_tolerance of BOTH true peaks — recalling
        # neither. (Under the Portilla et al. criteria a lighter blur, e.g.
        # size=9, still resolves two distinct significant peaks — the
        # min_bins/energy_frac test is more forgiving of a widened-but-
        # still-separated hump than the old prominence threshold was.)
        pred = uniform_filter1d(true, size=15)
        pred_peaks = find_spectral_peaks(FREQS, pred)
        assert len(pred_peaks) == 1  # confirms the blur actually merged them

        metrics, mask = peak_modality_metrics(FREQS, pred[np.newaxis, :], true[np.newaxis, :])

        assert mask[0]  # true spectrum is still multimodal
        assert metrics['Peak_Count_Pred_Mean'] == pytest.approx(1.0)
        assert metrics['Peak_Separation_Recall'] == pytest.approx(0.0)
        assert metrics['Peak_Height_RelError'] > 0.0

    def test_missed_secondary_peak_gives_finite_relative_error(self):
        """A small-but-real secondary peak that the model's prediction
        completely lacks must still produce a finite (not inf/nan)
        Peak_Height_RelError — no divide-by-zero blowup, since every
        detected peak's height is bounded below by the Portilla et al.
        (2009) energy_frac * E_total criterion > 0 by construction.
        """
        primary = _jonswap(FREQS, 2.0, 8.0)
        secondary = _jonswap(FREQS, 1.0, 16.0)
        true = primary + secondary
        pred = primary  # model produces only the primary peak

        assert len(find_spectral_peaks(FREQS, true)) == 2
        assert len(find_spectral_peaks(FREQS, pred)) == 1

        metrics, mask = peak_modality_metrics(FREQS, pred[np.newaxis, :], true[np.newaxis, :])

        assert mask[0]
        assert np.isfinite(metrics['Peak_Height_RelError'])
        assert 0.0 < metrics['Peak_Height_RelError'] < 2.0
        assert metrics['Peak_Separation_Recall'] == pytest.approx(0.5)


class TestPeakModalityMetricsPartitioned:
    """wind_sea/swell-conditioned breakdown (_windsea/_swell suffixes) and
    the per-partition Tm02 buckets — see utils/loss.py's
    SpectralKLDivergenceLoss-adjacent motivation in CLAUDE.md: a pooled
    Peak_Height_RelError/Peak_Separation_Recall/Tm02 blends two physically
    different predictability regimes (broad, fast-evolving wind-sea
    partitions vs narrow, persistent swell partitions) together, hiding
    which one a given loss change actually helped.
    """

    def test_identical_pred_and_true_zero_error_per_label(self):
        """pred == true exactly: whichever wind_sea/swell label(s) the true
        spectrum's two partitions get (derived independently via
        classify_partition, not assumed), every populated bucket must show
        perfect height/recall/Tm02 agreement, and the per-label counts must
        exactly account for both true partitions."""
        true = _bimodal(FREQS)
        pred = true.copy()

        windows = find_peak_windows(FREQS, true)
        assert len(windows) == 2  # precondition for this test to mean anything
        labels = [classify_partition(fp=FREQS[idx], S_obs_at_fp=true[idx])
                  for idx, _, _ in windows]

        metrics, mask = peak_modality_metrics(FREQS, pred[np.newaxis, :], true[np.newaxis, :])

        assert mask[0]
        assert metrics['Peak_windsea_n'] == labels.count('wind_sea')
        assert metrics['Peak_swell_n'] == labels.count('swell')
        assert metrics['Peak_windsea_n'] + metrics['Peak_swell_n'] == 2
        # pred == true exactly, so no partition's in-window energy ratio can
        # fall below the Tm02 masking threshold — every partition counted
        # above must also be counted here.
        assert metrics['Tm02_windsea_n'] == metrics['Peak_windsea_n']
        assert metrics['Tm02_swell_n'] == metrics['Peak_swell_n']

        for suffix in ('windsea', 'swell'):
            if metrics[f'Peak_{suffix}_n'] > 0:
                assert metrics[f'Peak_Height_RelError_{suffix}'] == pytest.approx(0.0, abs=1e-6)
                assert metrics[f'Peak_Separation_Recall_{suffix}'] == pytest.approx(1.0)
                assert metrics[f'Tm02_RMSE_{suffix}'] == pytest.approx(0.0, abs=1e-6)
                assert metrics[f'Tm02_Bias_{suffix}'] == pytest.approx(0.0, abs=1e-6)

    def test_missed_partition_excluded_from_tm02_but_counted_in_peak_n(self):
        """A partition the model's prediction carries essentially none of
        the true in-window energy for must be dropped from the Tm02 bucket
        (ill-conditioned near-zero mass) while still counting toward
        Peak_*_n — Peak_Separation_Recall (not Tm02) is what's supposed to
        register that specific miss."""
        primary = _jonswap(FREQS, 2.0, 8.0)     # fp=0.125 Hz -> wind_sea
        secondary = _jonswap(FREQS, 1.0, 16.0)  # fp=0.0625 Hz -> swell
        true = primary + secondary
        pred = primary  # model produces only the primary peak

        windows = find_peak_windows(FREQS, true)
        assert len(windows) == 2  # precondition

        metrics, mask = peak_modality_metrics(FREQS, pred[np.newaxis, :], true[np.newaxis, :])

        assert mask[0]
        assert metrics['Peak_windsea_n'] + metrics['Peak_swell_n'] == 2
        # The missed (swell) partition is excluded from the Tm02 buckets...
        assert metrics['Tm02_windsea_n'] + metrics['Tm02_swell_n'] < 2
        # ...but Peak_Separation_Recall for that label still reflects the miss.
        assert metrics['Peak_Separation_Recall_swell'] == pytest.approx(0.0)
        assert metrics['Peak_Separation_Recall_windsea'] == pytest.approx(1.0)
        assert np.isnan(metrics['Tm02_RMSE_swell'])
        assert np.isfinite(metrics['Tm02_RMSE_windsea'])
