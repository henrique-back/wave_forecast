"""
Tests for nn/spectrum_eval.py::compute_shape_final_metrics — the numpy
final-step shape scorer used to score forecasts that have no torch model
behind them (scripts/compare_physical_baseline.py).

  * parity: with m0_true=None it reproduces nn/evaluate.py's shape block on
    the same arrays;
  * identity: a perfect forecast scores zero error and full recall;
  * gamma* labels: with m0_true given, wind-sea/swell labels are computed on
    the physical spectrum, and nothing else in the peak panel changes;
  * a clipped/zero bin does not produce NaN in the Wasserstein term.
"""

import sys
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_spectral import _jonswap, FREQS
from nn.evaluate import evaluate
from nn.spectrum_eval import compute_shape_final_metrics
from utils.compute_hs import compute_shape
from utils.spectral_partitioning import gamma_star

F64 = FREQS.astype(np.float64)


def _spectra(n, seed=0):
    """Mix of unimodal and bimodal JONSWAP spectra with varied Hs/Tp."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        e = _jonswap(FREQS, rng.uniform(1.0, 3.0), rng.uniform(6.0, 14.0)).astype(np.float64)
        if i % 2:
            e = e + _jonswap(FREQS, rng.uniform(0.8, 2.0), rng.uniform(5.0, 7.0)).astype(np.float64)
        out.append(e)
    return np.stack(out)


class _StubShapeModel:
    """Minimal 'shape' model for evaluate(): forecasts the last input
    spectrum's shape shifted up by one bin, in log space."""
    target = "shape"

    def eval(self):
        return self

    def infer(self, src, freqs, lead_time, freq_means=None, shape_means=None, aux=None):
        last = (src[:, -1, :, 0] * freq_means).numpy().astype(np.float64)
        pred = compute_shape(np.roll(last, 1, axis=-1), freqs.numpy().astype(np.float64))
        pred = torch.from_numpy(np.log(np.clip(pred, 1e-12, None))).float()
        return pred.unsqueeze(1).expand(-1, lead_time, -1)


class TestParityWithEvaluate:
    def test_matches_evaluate_shape_block(self):
        n, seq_len, lead = 12, 3, 2
        inputs = _spectra(n * seq_len, seed=1).reshape(n, seq_len, -1)
        targets = _spectra(n * lead, seed=2).reshape(n, lead, -1)
        freq_means = torch.from_numpy(inputs.mean(axis=(0, 1))).float()
        X = torch.from_numpy(inputs / freq_means.numpy()).float().unsqueeze(-1)
        y = torch.from_numpy(compute_shape(targets, F64)).float()
        shape_means = torch.clamp(y.mean(dim=(0, 1)), min=1e-8)
        loader = DataLoader(TensorDataset(X, torch.zeros(n, seq_len, 0), y), batch_size=5)

        metrics, (y_pred, y_true, y_pers) = evaluate(
            _StubShapeModel(), loader, "cpu", torch.from_numpy(FREQS), lead_time=lead,
            freq_means=freq_means, shape_means=shape_means, return_arrays=True,
            compute_peak_metrics=True)
        ours = compute_shape_final_metrics(
            F64, np.exp(y_pred[:, -1].numpy()), np.exp(y_true[:, -1].numpy()),
            np.exp(y_pers[:, -1].numpy()), m0_true=None)

        for key, value in ours.items():
            if key not in metrics:
                continue  # keys the new scorer adds (e.g. *_pers, peak_fidelity_SS)
            np.testing.assert_allclose(np.asarray(value, dtype=float), np.asarray(metrics[key], dtype=float),
                                       rtol=1e-4, atol=1e-6, err_msg=key)


    def test_matches_evaluate_with_physical_labels(self):
        """With m0_true on the dataset (as _prepare_dataloaders attaches it),
        evaluate()'s peak panel matches the scorer given the same m0."""
        n, seq_len, lead = 12, 3, 2
        inputs = _spectra(n * seq_len, seed=1).reshape(n, seq_len, -1)
        targets = _spectra(n * lead, seed=2).reshape(n, lead, -1)
        freq_means = torch.from_numpy(inputs.mean(axis=(0, 1))).float()
        X = torch.from_numpy(inputs / freq_means.numpy()).float().unsqueeze(-1)
        y = torch.from_numpy(compute_shape(targets, F64)).float()
        shape_means = torch.clamp(y.mean(dim=(0, 1)), min=1e-8)
        dataset = TensorDataset(X, torch.zeros(n, seq_len, 0), y)
        dataset.m0_true = np.trapezoid(targets, F64, axis=-1)  # (n, lead)
        loader = DataLoader(dataset, batch_size=5)

        metrics, (y_pred, y_true, y_pers) = evaluate(
            _StubShapeModel(), loader, "cpu", torch.from_numpy(FREQS), lead_time=lead,
            freq_means=freq_means, shape_means=shape_means, return_arrays=True,
            compute_peak_metrics=True)
        ours = compute_shape_final_metrics(
            F64, np.exp(y_pred[:, -1].numpy()), np.exp(y_true[:, -1].numpy()),
            np.exp(y_pers[:, -1].numpy()), m0_true=dataset.m0_true[:, -1])

        for key in ("Peak_windsea_n", "Peak_swell_n", "Peak_Height_RelError_windsea",
                    "Peak_Height_RelError_swell", "Peak_Separation_Recall_swell",
                    "Tm02_RMSE_swell", "Peak_Count_True_Mean"):
            np.testing.assert_allclose(float(ours[key]), float(metrics[key]),
                                       rtol=1e-4, atol=1e-6, err_msg=key)


class TestIdentity:
    def test_perfect_forecast(self):
        truth = compute_shape(_spectra(6), F64)
        pers = compute_shape(_spectra(6, seed=3), F64)
        m = compute_shape_final_metrics(F64, truth, truth, pers, m0_true=np.ones(6))
        assert m["Shape_RMSE"] == 0.0
        assert abs(m["Shape_Wasserstein"]) < 1e-8
        assert m["Tm02_RMSE"] == 0.0
        assert m["Shape_SS"] == 1.0
        assert m["Peak_Separation_Recall"] == 1.0
        assert m["Peak_Height_RelError"] == 0.0


class TestGammaStarLabels:
    def test_labels_use_physical_units_and_nothing_else_changes(self):
        e = _jonswap(FREQS, 2.0, 10.0).astype(np.float64)
        p = int(np.argmax(e))
        shape = compute_shape(e[np.newaxis], F64)
        assert gamma_star(e[p], F64[p]) < 1.0 < gamma_star(shape[0, p], F64[p])  # precondition

        pred = compute_shape(np.roll(e, 1)[np.newaxis], F64)
        m0 = np.array([np.trapezoid(e, F64)])
        old = compute_shape_final_metrics(F64, pred, shape, shape, m0_true=None)
        new = compute_shape_final_metrics(F64, pred, shape, shape, m0_true=m0)

        assert (old["Peak_windsea_n"], old["Peak_swell_n"]) == (1, 0)
        assert (new["Peak_windsea_n"], new["Peak_swell_n"]) == (0, 1)
        for key in ("Peak_Count_True_Mean", "Peak_Count_Pred_Mean", "Peak_Separation_Recall",
                    "Peak_Height_RelError", "Tm02_RMSE", "Shape_RMSE", "Shape_Wasserstein"):
            np.testing.assert_allclose(new[key], old[key], rtol=1e-10, err_msg=key)


class TestZeroBins:
    def test_zero_bin_gives_finite_wasserstein(self):
        truth = compute_shape(_spectra(4), F64)
        pred = truth.copy()
        pred[:, :5] = 0.0  # e.g. a clipped negative AR bin
        pred = compute_shape(pred, F64)
        m = compute_shape_final_metrics(F64, pred, truth, truth, m0_true=np.ones(4))
        assert np.isfinite(m["Shape_Wasserstein"])
