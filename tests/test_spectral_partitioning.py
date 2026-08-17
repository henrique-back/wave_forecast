"""
Tests for utils/spectral_partitioning.py::find_peak_windows — the
trough-to-trough partition window (left_idx, right_idx) that
utils.loss.SoftPeakHeightLoss needs per detected peak, exposed alongside
find_significant_peaks' own peak-index list rather than a fixed bin-radius
around each peak (see the class docstring in utils/loss.py for why a fixed
radius is wrong: swell partitions are narrow, wind-sea partitions are wide).

Reuses the JONSWAP generator and frequency grid from tests/test_spectral.py,
same convention as tests/test_spectral_peaks.py.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_spectral import _jonswap, FREQS
from utils.spectral_partitioning import find_peak_windows, find_significant_peaks, _trough


def _bimodal(freqs):
    """Two comparably-sized, well-separated peaks (Tp=7s and Tp=14s) — same
    fixture as tests/test_spectral_peaks.py."""
    return _jonswap(freqs, 2.0, 7.0) + _jonswap(freqs, 2.0, 14.0)


class TestFindPeakWindows:
    def test_single_peak_window_spans_whole_spectrum(self):
        """No second peak to create an internal trough, so the one
        surviving peak's partition should run edge to edge."""
        spectrum = _jonswap(FREQS, 2.0, 10.0)
        windows = find_peak_windows(FREQS, spectrum)

        assert len(windows) == 1
        peak_idx, left_idx, right_idx = windows[0]
        assert left_idx == 0
        assert right_idx == len(FREQS) - 1
        assert left_idx <= peak_idx <= right_idx

    def test_two_peaks_share_the_trough_as_a_common_boundary(self):
        """Adjacent partitions must be contiguous, never overlapping — the
        right_idx of the lower-frequency peak's window must equal the
        left_idx of the higher-frequency peak's window (both are the same
        trough bin), which is what guarantees SoftPeakHeightLoss's windows
        for a bimodal (wind-sea + swell) spectrum never bleed into each
        other."""
        spectrum = _bimodal(FREQS)
        windows = find_peak_windows(FREQS, spectrum)

        assert len(windows) == 2
        (peak1, left1, right1), (peak2, left2, right2) = windows
        assert peak1 < peak2  # ascending frequency order
        assert left1 == 0
        assert right2 == len(FREQS) - 1
        assert right1 == left2  # shared trough boundary, no gap, no overlap

        # Sanity: the shared boundary really is the minimum between the peaks.
        expected_trough = _trough(spectrum, peak1, peak2)
        assert right1 == expected_trough

    def test_windows_bracket_their_own_peak_index(self):
        spectrum = _bimodal(FREQS)
        windows = find_peak_windows(FREQS, spectrum)
        for peak_idx, left_idx, right_idx in windows:
            assert left_idx <= peak_idx <= right_idx
            assert 0 <= left_idx <= right_idx <= len(FREQS) - 1

    def test_peak_indices_match_find_significant_peaks(self):
        """find_peak_windows must not silently disagree with
        find_significant_peaks about which peaks survive the four Portilla
        criteria — it's meant to be the same list, just enriched with
        window bounds."""
        spectrum = _bimodal(FREQS)
        peak_idxs = [p for p, _, _ in find_peak_windows(FREQS, spectrum)]
        assert peak_idxs == find_significant_peaks(FREQS, spectrum)

    def test_zero_spectrum_returns_empty_without_raising(self):
        spectrum = np.zeros_like(FREQS)
        assert find_peak_windows(FREQS, spectrum) == []

    def test_flat_spectrum_returns_empty_without_raising(self):
        spectrum = np.ones_like(FREQS)
        assert find_peak_windows(FREQS, spectrum) == []

    def test_spurious_peak_filtered_out_has_no_window(self):
        """A tiny secondary bump too low-energy to pass criterion 2 must be
        absent from find_peak_windows exactly as it's absent from
        find_significant_peaks — the window list is gated by the same four
        criteria, not a superset of every raw local maximum."""
        primary = _jonswap(FREQS, 2.0, 10.0)
        tiny_bump = 1e-4 * _jonswap(FREQS, 0.05, 18.0)
        spectrum = primary + tiny_bump

        significant = find_significant_peaks(FREQS, spectrum, energy_frac=0.05)
        windows = find_peak_windows(FREQS, spectrum, energy_frac=0.05)

        assert len(significant) == 1
        assert len(windows) == 1
