"""
Tests for nn/prepare_dmd.py::compute_dmd_features — DMD-derived
(growth_rate, frequency, amplitude) features used by the 'dmd' aux_set
(nn/channels.py) to give the encoder direct, physically-grounded
information about whether the currently-observed wave systems are growing
or decaying, rather than requiring the Transformer to infer that
implicitly from raw historical spectra.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nn.prepare_dmd import compute_dmd_features
from nn.channels import AUX_CHANNEL_SETS, AUX_NORM_MODES


def _decaying_mode(num_freqs, t, growth_rate, seed):
    """A real (non-oscillating) DMD mode: one spatial pattern, exponentially
    growing/decaying in time. Correctly rank-1 for a REAL eigenvalue."""
    rng = np.random.default_rng(seed)
    pattern = rng.random(num_freqs) + 0.5
    return np.outer(np.exp(growth_rate * t), pattern)


def _oscillating_mode(num_freqs, t, growth_rate, freq, seed):
    """A complex-conjugate-pair DMD mode: needs a genuine 2-D real spatial
    subspace (Re(v), Im(v) of the complex eigenvector), NOT a single
    spatial pattern times cos(freq*t) — that construction is actually
    rank-1 in space and does NOT satisfy linear dynamics x(t+1)=Ax(t) for
    a fixed A (the ratio x(t+1)/x(t) varies with t through the cosine's
    sign changes), so DMD cannot recover it. Confirmed empirically while
    writing this test: a single-pattern-times-cosine construction produced
    unrecoverable growth/freq values; this two-pattern construction is the
    correct way to synthesize an oscillating DMD mode."""
    rng = np.random.default_rng(seed)
    pattern_re = rng.random(num_freqs) + 0.5
    pattern_im = rng.random(num_freqs) + 0.5
    envelope = np.exp(growth_rate * t)
    return (np.outer(envelope * np.cos(freq * t), pattern_re)
            - np.outer(envelope * np.sin(freq * t), pattern_im))


class TestChannelRegistryConsistency:
    def test_dmd_column_count_matches_default_n_modes(self):
        from nn.prepare_dmd import DEFAULT_N_MODES
        assert len(AUX_CHANNEL_SETS['dmd']) == 3 * DEFAULT_N_MODES

    def test_every_dmd_column_has_a_norm_mode(self):
        for name in AUX_CHANNEL_SETS['dmd']:
            assert name in AUX_NORM_MODES
            assert AUX_NORM_MODES[name] == 'zscore'


class TestSyntheticModeRecovery:
    """The standard way to validate a DMD implementation: construct a
    signal from known growth rate(s)/frequency(ies), confirm they come
    back out."""

    def test_recovers_single_decaying_mode(self):
        num_freqs, seq_len = 10, 20
        t = np.arange(seq_len)
        window = _decaying_mode(num_freqs, t, growth_rate=-0.05, seed=0)
        feats = compute_dmd_features(window[np.newaxis], n_modes=2)

        assert feats[0, 0] == pytest.approx(-0.05, abs=1e-3)  # growth
        assert feats[0, 1] == pytest.approx(0.0, abs=1e-3)    # freq

    def test_recovers_growing_and_decaying_modes_together(self):
        """One real decaying mode + one complex oscillating mode, summed —
        both must be recovered, sorted by amplitude, without either being
        lost or the oscillating mode being double-counted as two slots."""
        num_freqs, seq_len = 10, 30
        t = np.arange(seq_len)
        signal = (_decaying_mode(num_freqs, t, growth_rate=-0.05, seed=0)
                  + _oscillating_mode(num_freqs, t, growth_rate=0.03, freq=0.4, seed=1))
        feats = compute_dmd_features(signal[np.newaxis], n_modes=3)

        recovered = [(feats[0, 3*k], feats[0, 3*k+1]) for k in range(3)]

        def _has_close_match(target, tol=1e-2):
            return any(abs(g - target[0]) < tol and abs(f - target[1]) < tol
                       for g, f in recovered)

        assert _has_close_match((-0.05, 0.0)), recovered
        assert _has_close_match((0.03, 0.4)), recovered

    def test_batch_of_samples_recovered_independently(self):
        num_freqs, seq_len = 8, 20
        t = np.arange(seq_len)
        window_a = _decaying_mode(num_freqs, t, growth_rate=-0.08, seed=2)
        window_b = _decaying_mode(num_freqs, t, growth_rate=0.06, seed=3)
        batch = np.stack([window_a, window_b])

        feats = compute_dmd_features(batch, n_modes=2)
        assert feats.shape == (2, 6)
        assert feats[0, 0] == pytest.approx(-0.08, abs=1e-3)
        assert feats[1, 0] == pytest.approx(0.06, abs=1e-3)


class TestEdgeCases:
    def test_degenerate_flat_window_does_not_crash(self):
        num_freqs, seq_len = 10, 20
        window = np.ones((1, seq_len, num_freqs))
        feats = compute_dmd_features(window, n_modes=4)
        assert np.all(np.isfinite(feats))

    def test_zero_window_does_not_crash(self):
        num_freqs, seq_len = 10, 20
        window = np.zeros((1, seq_len, num_freqs))
        feats = compute_dmd_features(window, n_modes=4)
        assert np.all(np.isfinite(feats))

    def test_output_shape_matches_n_modes(self):
        num_freqs, seq_len, n_modes = 12, 16, 5
        rng = np.random.default_rng(0)
        window = rng.random((3, seq_len, num_freqs))
        feats = compute_dmd_features(window, n_modes=n_modes)
        assert feats.shape == (3, 3 * n_modes)
