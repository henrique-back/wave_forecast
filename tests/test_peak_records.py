"""
Tests for utils/peak_records.py: pooling its per-sample statistics must give
exactly what peak_modality_metrics + _compute_val_score('peak_fidelity') give
on the same batch, for the whole batch and for any subset of it.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_spectral import _jonswap, FREQS
from nn.optimization import _compute_val_score
from utils.peak_records import peak_stats, pooled_peak_fidelity
from utils.spectral_peaks import peak_modality_metrics


def _batch(n, seed):
    """Physical spectra with 1-3 JONSWAP systems plus noise; pred perturbs true."""
    rng = np.random.default_rng(seed)
    true, pred = [], []
    for _ in range(n):
        k = rng.integers(1, 4)
        s = sum(_jonswap(FREQS, rng.uniform(0.5, 3.0), rng.uniform(4.0, 16.0)) for _ in range(k))
        true.append(s)
        p = s * rng.lognormal(0.0, 0.3, size=s.shape)
        if rng.random() < 0.3:
            p = p + _jonswap(FREQS, rng.uniform(0.5, 2.0), rng.uniform(4.0, 16.0))
        pred.append(p)
    return np.array(pred), np.array(true)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_pooled_matches_peak_modality_metrics(seed):
    pred, true = _batch(60, seed)
    stats, n_true, _ = peak_stats(FREQS, pred, true)
    pooled = pooled_peak_fidelity(stats[:, 0])
    ref, mask = peak_modality_metrics(FREQS, pred, true)

    assert np.array_equal(n_true >= 2, mask)
    assert pooled['precision'] == pytest.approx(ref['Peak_Separation_Precision'], nan_ok=True)
    assert pooled['peak_fidelity'] == pytest.approx(_compute_val_score(ref, 'peak_fidelity'))
    for label, suffix in (('wind_sea', 'windsea'), ('swell', 'swell')):
        per = pooled_peak_fidelity(stats[:, 0], label=label)
        assert per['recall'] == pytest.approx(ref[f'Peak_Separation_Recall_{suffix}'], nan_ok=True)
        assert per['rel_err'] == pytest.approx(ref[f'Peak_Height_RelError_{suffix}'], nan_ok=True)
        assert per['n_true'] == ref[f'Peak_{suffix}_n']


def test_subset_weights_match_rescoring_the_subset():
    pred, true = _batch(50, 3)
    stats, _, _ = peak_stats(FREQS, pred, true)
    sel = np.arange(50) % 3 == 0
    ref, _ = peak_modality_metrics(FREQS, pred[sel], true[sel])
    got = pooled_peak_fidelity(stats[:, 0], weights=sel.astype(float))
    assert got['peak_fidelity'] == pytest.approx(_compute_val_score(ref, 'peak_fidelity'))


def test_bootstrap_multiplicities_match_duplicated_batch():
    pred, true = _batch(30, 4)
    stats, _, _ = peak_stats(FREQS, pred, true)
    idx = np.random.default_rng(0).integers(0, 30, size=30)
    ref, _ = peak_modality_metrics(FREQS, pred[idx], true[idx])
    counts = np.bincount(idx, minlength=30).astype(float)
    got = pooled_peak_fidelity(stats[:, 0], weights=np.stack([counts, counts]))
    assert got['peak_fidelity'].shape == (2,)
    assert got['peak_fidelity'][0] == pytest.approx(_compute_val_score(ref, 'peak_fidelity'))


def test_bands_partition_the_all_group():
    pred, true = _batch(40, 5)
    stats, _, _ = peak_stats(FREQS, pred, true, band_edges=(0.08, 0.125, 0.2))
    assert stats.shape[1] == 5
    np.testing.assert_allclose(stats[:, 1:].sum(axis=1), stats[:, 0])


def test_identical_forecast_scores_one():
    _, true = _batch(20, 6)
    stats, _, _ = peak_stats(FREQS, true.copy(), true)
    assert pooled_peak_fidelity(stats[:, 0])['peak_fidelity'] == pytest.approx(1.0)


def test_no_true_peaks_is_nan():
    flat = np.ones((3, len(FREQS)))
    stats, n_true, _ = peak_stats(FREQS, flat, flat)
    assert (n_true == 0).all()
    assert np.isnan(pooled_peak_fidelity(stats[:, 0])['peak_fidelity'])
