"""Sanity tests for the linear AR baseline (utils/linear_baseline.py)."""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.linear_baseline import (
    _split_train_val_test,
    fit_linear_ar,
    fit_linear_ar_from_density,
    _rollout,
    _make_windows,
    forecast_coeffs,
    evaluate_coeffs,
    evaluate_linear_ar,
)


class TestSplitFractions:
    def test_matches_optimization_split(self):
        """Must reproduce nn/optimization.py::_prepare_dataloaders's exact
        70/15/15 split fractions, since this baseline's test slice needs to
        be the same time range a transformer experiment was scored on."""
        n = 1000
        df = pd.DataFrame(np.zeros((n, 3)), columns=['0.02', '0.05', '0.1'])
        train, val, test = _split_train_val_test(df)
        assert len(train) == int(0.7 * n)
        assert len(train) + len(val) == int(0.85 * n)
        assert len(train) + len(val) + len(test) == n


class TestFitAndRollout:
    def test_recovers_known_ar1_coefficient(self):
        """x_{t+1} = 0.5 * x_t (no noise) — fit_linear_ar must recover the
        0.5 weight and ~0 intercept for a single-column series."""
        rng = np.random.default_rng(0)
        x = np.zeros(2000)
        x[0] = 1.0
        for t in range(1, len(x)):
            x[t] = 0.5 * x[t - 1] + rng.normal(scale=1e-6)
        coeffs = fit_linear_ar(x, order=1, ridge=1e-8)
        assert coeffs.shape == (1, 2)
        assert abs(coeffs[0, 0]) < 1e-3          # intercept ~ 0
        assert abs(coeffs[0, 1] - 0.5) < 1e-3    # lag weight ~ 0.5

    def test_rollout_forecasts_known_decay(self):
        """Given the AR(1) coefficient above, rolling out from x_t=2.0 for 3
        steps should give 1.0, 0.5, 0.25."""
        coeffs = np.array([[0.0, 0.5]])  # (num_cols=1, order+1=2)
        last_windows = np.array([[[2.0]]])  # (num_samples=1, order=1, num_cols=1)
        forecasts = _rollout(coeffs, last_windows, lead_time=3)
        assert forecasts.shape == (1, 3, 1)
        np.testing.assert_allclose(forecasts[0, :, 0], [1.0, 0.5, 0.25], atol=1e-8)

    def test_rollout_renormalizes_shape_target(self):
        """When freqs is given, each rolled-out step must integrate to 1
        (unit-area shape), even though the underlying per-bin AR predictions
        have no reason to sum to exactly 1 on their own."""
        freqs = np.linspace(0.02, 0.5, 8)
        num_cols = len(freqs)
        # Arbitrary AR(1) coefficients per bin (not fit from real data —
        # just needs to produce some non-degenerate per-bin prediction).
        rng = np.random.default_rng(1)
        coeffs = np.column_stack([np.zeros(num_cols), rng.uniform(0.3, 0.9, num_cols)])
        last_windows = rng.uniform(0.5, 1.5, size=(5, 1, num_cols))
        forecasts = _rollout(coeffs, last_windows, lead_time=4, freqs=freqs)
        integrals = np.trapezoid(forecasts, freqs, axis=2)  # (num_samples, lead_time)
        np.testing.assert_allclose(integrals, 1.0, atol=1e-4)


class TestMakeWindows:
    def test_shapes_and_indexing(self):
        series = np.arange(20, dtype=np.float64)[:, np.newaxis]  # (20, 1)
        order, lead_time = 5, 3
        windows, targets = _make_windows(series, order, lead_time)
        num_samples = 20 - order - lead_time + 1
        assert windows.shape == (num_samples, order, 1)
        assert targets.shape == (num_samples, lead_time, 1)
        # Sample 0: window = [0,1,2,3,4], target = [5,6,7]
        np.testing.assert_allclose(windows[0, :, 0], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(targets[0, :, 0], [5, 6, 7])
        # Last sample's target must reach exactly the end of the series.
        np.testing.assert_allclose(targets[-1, -1, 0], 19)


def _synthetic_density(n_timesteps=500, seed=0):
    """A small multi-frequency-bin dataset with genuine AR(1)-like temporal
    structure per bin, positive-valued (so it's a plausible spectral density)."""
    rng = np.random.default_rng(seed)
    freqs = np.linspace(0.02, 0.485, 12)
    num_freqs = len(freqs)
    base = np.array([1.0 + 3.0 * np.exp(-((f - 0.1) ** 2) / 0.01) for f in freqs])
    density = np.zeros((n_timesteps, num_freqs))
    density[0] = base
    for t in range(1, n_timesteps):
        density[t] = 0.85 * density[t - 1] + 0.15 * base + rng.normal(scale=0.02, size=num_freqs)
    density = np.clip(density, 1e-6, None)
    columns = [str(f) for f in freqs]
    return pd.DataFrame(density, columns=columns), freqs


class TestEvaluateLinearAR:
    @pytest.mark.parametrize("target", ["hs", "density", "shape"])
    def test_returns_finite_metrics(self, target):
        density, freqs = _synthetic_density()
        metrics = evaluate_linear_ar(density, freqs, seq_len=6, lead_time=4, target=target)
        assert len(metrics['per_step_RMSE']) == 4
        for key in ('per_step_RMSE', 'per_step_RMSE_pers', 'per_step_Bias'):
            assert all(np.isfinite(v) for v in metrics[key])
        if target in ('hs', 'density'):
            assert np.isfinite(metrics['Hs_SS'])
        if target == 'hs':
            for key in ('RMSE', 'CC', 'R2', 'overall_SS', 'Hs_MAPE'):
                assert np.isfinite(metrics[key])
        if target == 'density':
            for key in ('Hs_RMSE', 'Tm02_RMSE', 'Shape_RMSE', 'SI_mean'):
                assert np.isfinite(metrics[key])
            assert len(metrics['SI_per_bin']) == len(freqs)
        if target == 'shape':
            # No Hs_SS for a pure shape target — see _compute_metrics docstring.
            assert 'Hs_SS' not in metrics
            assert metrics['Shape_RMSE'] == metrics['per_step_RMSE'][-1]
            assert metrics['Shape_SS'] == metrics['per_step_SS'][-1]

    def test_invalid_target_raises(self):
        density, freqs = _synthetic_density()
        with pytest.raises(ValueError):
            evaluate_linear_ar(density, freqs, seq_len=6, lead_time=4, target='bogus')


class TestFitEvaluateSplit:
    """fit_linear_ar_from_density + evaluate_coeffs/forecast_coeffs must
    behave identically to the fused evaluate_linear_ar convenience wrapper —
    this is the split used by scripts/train_linear_baseline.py (fit once,
    save coeffs) and scripts/compare_versions.py (load coeffs, never refit)."""

    def test_matches_fused_wrapper(self):
        density, freqs = _synthetic_density()
        fused = evaluate_linear_ar(density, freqs, seq_len=6, lead_time=4, target='density')

        coeffs = fit_linear_ar_from_density(density, freqs, seq_len=6, target='density')
        split = evaluate_coeffs(density, freqs, coeffs, seq_len=6, lead_time=4, target='density')

        assert fused['per_step_RMSE'] == pytest.approx(split['per_step_RMSE'])
        assert fused['Hs_SS'] == pytest.approx(split['Hs_SS'])

    def test_forecast_coeffs_returns_matching_shapes(self):
        density, freqs = _synthetic_density()
        coeffs = fit_linear_ar_from_density(density, freqs, seq_len=6, target='density')
        pred, true, pers = forecast_coeffs(density, freqs, coeffs, seq_len=6, lead_time=4, target='density')
        assert pred.shape == true.shape == pers.shape
        assert pred.shape[1] == 4
        assert pred.shape[2] == len(freqs)

    def test_val_and_test_splits_differ(self):
        """A sanity check that eval_split actually selects different data —
        not a claim about which score is better."""
        density, freqs = _synthetic_density(n_timesteps=800)
        coeffs = fit_linear_ar_from_density(density, freqs, seq_len=6, target='hs')
        val_metrics = evaluate_coeffs(density, freqs, coeffs, seq_len=6, lead_time=4, target='hs', eval_split='val')
        test_metrics = evaluate_coeffs(density, freqs, coeffs, seq_len=6, lead_time=4, target='hs', eval_split='test')
        assert val_metrics['per_step_RMSE'] != test_metrics['per_step_RMSE']

    def test_invalid_eval_split_raises(self):
        density, freqs = _synthetic_density()
        coeffs = fit_linear_ar_from_density(density, freqs, seq_len=6, target='hs')
        with pytest.raises(ValueError):
            evaluate_coeffs(density, freqs, coeffs, seq_len=6, lead_time=4, target='hs', eval_split='bogus')
