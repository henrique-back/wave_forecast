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
from utils.spectral_peaks import find_spectral_peaks, peak_modality_metrics


def _bimodal(freqs):
    """Two comparably-sized, well-separated peaks (Tp=7s and Tp=14s)."""
    return _jonswap(freqs, 2.0, 7.0) + _jonswap(freqs, 2.0, 14.0)


class TestFindSpectralPeaks:
    def test_single_jonswap_has_one_peak(self):
        spectrum = _jonswap(FREQS, 2.0, 10.0)
        peaks = find_spectral_peaks(spectrum)
        assert len(peaks) == 1

    def test_sum_of_two_separated_jonswaps_has_two_peaks(self):
        spectrum = _bimodal(FREQS)
        peaks = find_spectral_peaks(spectrum)
        assert len(peaks) == 2

    def test_zero_spectrum_returns_empty_without_raising(self):
        spectrum = np.zeros_like(FREQS)
        peaks = find_spectral_peaks(spectrum)
        assert len(peaks) == 0

    def test_flat_spectrum_returns_empty_without_raising(self):
        spectrum = np.ones_like(FREQS)
        peaks = find_spectral_peaks(spectrum)
        assert len(peaks) == 0


class TestPeakModalityMetrics:
    def test_identical_pred_and_true_mixed_batch(self):
        """One unimodal + one bimodal sample, pred == true exactly."""
        unimodal = _jonswap(FREQS, 2.0, 10.0)
        bimodal = _bimodal(FREQS)
        true = np.stack([unimodal, bimodal])
        pred = true.copy()

        metrics, mask = peak_modality_metrics(pred, true)

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
        true_peaks = find_spectral_peaks(true)
        assert len(true_peaks) == 2  # precondition for this test to mean anything

        # size=9 uniform smoothing merges the two peaks into a single hump
        # that sits near, but not within bin_tolerance of, the first true
        # peak and far from the second — recalling exactly one of two.
        pred = uniform_filter1d(true, size=9)
        pred_peaks = find_spectral_peaks(pred)
        assert len(pred_peaks) == 1  # confirms the blur actually merged them

        metrics, mask = peak_modality_metrics(pred[np.newaxis, :], true[np.newaxis, :])

        assert mask[0]  # true spectrum is still multimodal
        assert metrics['Peak_Count_Pred_Mean'] == pytest.approx(1.0)
        assert metrics['Peak_Separation_Recall'] == pytest.approx(0.5)
        assert metrics['Peak_Height_RelError'] > 0.0

    def test_missed_secondary_peak_gives_finite_relative_error(self):
        """A small-but-real secondary peak that the model's prediction
        completely lacks must still produce a finite (not inf/nan)
        Peak_Height_RelError — no divide-by-zero blowup, since every
        detected peak's height is bounded below by prominence_frac *
        spectrum.max() > 0 by construction.
        """
        primary = _jonswap(FREQS, 2.0, 8.0)
        secondary = _jonswap(FREQS, 1.0, 16.0)
        true = primary + secondary
        pred = primary  # model produces only the primary peak

        assert len(find_spectral_peaks(true)) == 2
        assert len(find_spectral_peaks(pred)) == 1

        metrics, mask = peak_modality_metrics(pred[np.newaxis, :], true[np.newaxis, :])

        assert mask[0]
        assert np.isfinite(metrics['Peak_Height_RelError'])
        assert 0.0 < metrics['Peak_Height_RelError'] < 2.0
        assert metrics['Peak_Separation_Recall'] == pytest.approx(0.5)
