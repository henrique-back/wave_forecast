"""
Tests for the helpers behind the physical-model baseline
(scripts/compare_physical_baseline.py, scripts/fetch_gefs_reforecast.py):

  * start-time mapping: buoy row k of a start time maps to transformer sample
    j = k - val_end - L + 1 and AR sample i = k - val_end - p + 1, whose last
    input row is k and whose final target row is k + lead;
  * band helpers: the band excludes buoy bins below the model grid, the
    interpolation refuses to extrapolate, and band shapes have unit area;
  * fetch helpers: direction integration and WW3 time decoding.
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.prepare_x import prepare_X
from nn.prepare_y import prepare_y
from utils.linear_baseline import _make_windows
from scripts.compare_physical_baseline import band_mask, band_shape, interp_to, eligible_rows

NDBC_FREQS = np.array([0.02, 0.0325] + list(np.round(np.arange(0.0375, 0.0926, 0.005), 4))
                      + list(np.round(np.arange(0.10, 0.351, 0.01), 4))
                      + list(np.round(np.arange(0.365, 0.486, 0.02), 4)))


def _row_coded_frame(n=200, n_freqs=5):
    """Hourly frame whose value in every bin is its own row number + 1."""
    index = pd.date_range("2017-01-01", periods=n, freq="h")
    values = np.repeat(np.arange(1, n + 1, dtype=float)[:, None], n_freqs, axis=1)
    return pd.DataFrame(values, index=index, columns=[f"{0.05 * (c + 1):.2f}" for c in range(n_freqs)])


class TestStartTimeMapping:
    @pytest.mark.parametrize("seq_len,order,lead", [(12, 48, 12), (48, 12, 48), (24, 24, 24)])
    def test_rows(self, seq_len, order, lead):
        frame = _row_coded_frame(n=400)
        val_end = 200                                # rows 200-399 form the test split
        test = frame.iloc[val_end:]
        starts = frame.index[[val_end + 50, val_end + 70]]
        k, ok = eligible_rows(frame.index, starts, val_end, max(seq_len, order), lead)
        assert ok.all()

        X = prepare_X([test], seq_length=seq_len, lead_time=lead).numpy()
        y = prepare_y(test, seq_length=seq_len, lead_time=lead, target="density").numpy()
        j = k - val_end - seq_len + 1
        np.testing.assert_array_equal(X[j, -1, 0, 0], k + 1)          # last input row is k
        np.testing.assert_array_equal(y[j, -1, 0], k + lead + 1)      # final target row is k + lead

        windows, targets = _make_windows(test.to_numpy(), order, lead)
        i = k - val_end - order + 1
        np.testing.assert_array_equal(windows[i, -1, 0], k + 1)
        np.testing.assert_array_equal(targets[i, -1, 0], k + lead + 1)

    def test_ineligible_start_times(self):
        frame = _row_coded_frame()
        val_end = 100
        starts = frame.index[[val_end + 5, len(frame) - 3]] # window too short / lead too long
        _, ok = eligible_rows(frame.index, starts, val_end, 12, 12)
        assert not ok.any()


class TestBand:
    def test_band_excludes_bins_below_the_model_grid(self):
        band = band_mask(NDBC_FREQS, np.array([0.035, 0.9635]))
        assert NDBC_FREQS[band][0] == 0.0375 and band.sum() == len(NDBC_FREQS) - 2

    def test_interp_refuses_to_extrapolate(self):
        with pytest.raises(ValueError):
            interp_to(np.ones((1, 3)), np.array([0.035, 0.1, 0.4]), NDBC_FREQS)

    def test_band_shape_has_unit_area(self):
        band = band_mask(NDBC_FREQS, np.array([0.035, 0.9635]))
        spectra = np.abs(np.random.default_rng(0).normal(size=(3, len(NDBC_FREQS))))
        spectra[0, 10] = -1.0                                        # clipped, not propagated
        shapes = band_shape(spectra, NDBC_FREQS, band)
        np.testing.assert_allclose(np.trapezoid(shapes, NDBC_FREQS[band], axis=1), 1.0)
        assert (shapes >= 0).all()


class TestFetchHelpers:
    def test_integrate_direction_and_time_decoding(self):
        pytest.importorskip("h5py")
        from scripts.fetch_gefs_reforecast import integrate_direction, decode_times

        np.testing.assert_allclose(integrate_direction(np.full((2, 4, 36), 3.0), 36), 3.0 * 2 * np.pi)
        days = np.array([10000.125, 10000.25])                       # 03:00 and 06:00
        times = decode_times(days)
        assert list(times.hour) == [3, 6] and (times.minute == 0).all()
