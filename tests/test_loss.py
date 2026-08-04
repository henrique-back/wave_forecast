"""
Tests for utils.loss.DirectionalLoss — the weighted composite loss over
spectral density + directional wave parameters proposed in Meta 3 of the
2026-07-24 meeting doc (deliverable 3.5: "test a first version of the
weighted loss").
"""
import os
import sys

import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.loss import DirectionalLoss, SpectralWassersteinLoss, _FULL_CHANNELS
from tests.test_spectral import FREQS as SPECTRAL_FREQS


def _make_sample(density, alpha1_deg, alpha2_deg, r1, r2=0.5, num_freqs=4):
    """(num_freqs, 7) tensor in DirectionalLoss's expected channel order."""
    a1 = np.radians(alpha1_deg)
    a2 = np.radians(alpha2_deg)
    return torch.tensor([
        [density, np.sin(a1), np.cos(a1), np.sin(a2), np.cos(a2), r1, r2]
        for _ in range(num_freqs)
    ], dtype=torch.float32)


class TestDirectionalLossChannelOrder:
    def test_matches_nn_channels_full(self):
        """_FULL_CHANNELS must stay in sync with nn.channels.CHANNEL_SETS
        ['full'] — hardcoded locally to avoid a circular import (utils must
        not depend on nn), so this test is the drift guard."""
        from nn.channels import CHANNEL_SETS
        assert _FULL_CHANNELS == CHANNEL_SETS['full']


class TestDirectionalLossBasics:
    def test_zero_loss_for_perfect_prediction(self):
        loss_fn = DirectionalLoss()
        true = _make_sample(1.5, 45.0, 200.0, 0.6)
        total, components = loss_fn(true, true)
        assert total.item() == pytest.approx(0.0, abs=1e-6)
        for v in components.values():
            assert v.item() == pytest.approx(0.0, abs=1e-6)

    def test_total_equals_weighted_sum_of_components(self):
        loss_fn = DirectionalLoss(lambda_E=2.0, lambda_alpha1=0.5,
                                   lambda_alpha2=0.1, lambda_r=1.5)
        true = _make_sample(1.5, 45.0, 200.0, 0.6)
        pred = _make_sample(2.0, 60.0, 210.0, 0.4)
        total, c = loss_fn(true, pred)
        expected = (2.0 * c['L_E'] + 0.5 * c['L_alpha1']
                    + 0.1 * c['L_alpha2'] + 1.5 * c['L_r'])
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_lambda_zero_removes_component_contribution(self):
        """Setting a lambda to 0 must not change total even if that
        component's prediction is wildly wrong."""
        true = _make_sample(1.5, 45.0, 200.0, 0.6)
        bad_r_pred = _make_sample(1.5, 45.0, 200.0, 99.0)  # only r1 is wrong

        loss_with_r = DirectionalLoss(lambda_E=1.0, lambda_alpha1=1.0,
                                       lambda_alpha2=1.0, lambda_r=1.0)
        loss_without_r = DirectionalLoss(lambda_E=1.0, lambda_alpha1=1.0,
                                          lambda_alpha2=1.0, lambda_r=0.0)

        total_with_r, _ = loss_with_r(bad_r_pred, true)
        total_without_r, _ = loss_without_r(bad_r_pred, true)

        assert total_with_r.item() > 1.0  # r1 error dominates
        assert total_without_r.item() == pytest.approx(0.0, abs=1e-6)


class TestDirectionalLossAngularWraparound:
    """The core value proposition of sin/cos over raw-degree MSE: physically
    adjacent angles across the 0/360 boundary must produce a small loss."""

    def test_wraparound_gives_small_loss_not_large(self):
        loss_fn = DirectionalLoss()
        true = _make_sample(1.0, 1.0, 180.0, 0.5)      # alpha1 = 1 deg
        pred = _make_sample(1.0, 359.0, 180.0, 0.5)     # alpha1 = 359 deg

        # Physically 2 degrees apart, not 358.
        _, components = loss_fn(pred, true)
        # A naive raw-degree MSE would give (359-1)^2 = 128164; sin/cos
        # encoding must stay tiny for a 2-degree true separation.
        assert components['L_alpha1'].item() < 0.01

    def test_wraparound_loss_matches_small_true_separation(self):
        """1 deg vs 359 deg (sin/cos loss) should be close to 1 deg vs 3 deg
        (a small, non-wrapping separation of the same 2-degree magnitude)."""
        loss_fn = DirectionalLoss()

        true_wrap = _make_sample(1.0, 1.0, 180.0, 0.5)
        pred_wrap = _make_sample(1.0, 359.0, 180.0, 0.5)
        _, c_wrap = loss_fn(pred_wrap, true_wrap)

        true_nowrap = _make_sample(1.0, 1.0, 180.0, 0.5)
        pred_nowrap = _make_sample(1.0, 3.0, 180.0, 0.5)
        _, c_nowrap = loss_fn(pred_nowrap, true_nowrap)

        assert c_wrap['L_alpha1'].item() == pytest.approx(
            c_nowrap['L_alpha1'].item(), rel=0.05
        )


class TestDirectionalLossFreqWeighting:
    def test_freq_weights_change_result_when_nonuniform(self):
        loss_fn = DirectionalLoss()
        num_freqs = 4
        true = _make_sample(1.5, 45.0, 200.0, 0.6, num_freqs=num_freqs)
        pred = true.clone()
        pred[0, 0] += 1.0  # perturb density only at the first bin

        flat_total, _ = loss_fn(pred, true)

        # Weight the first bin heavily relative to the rest.
        weights = torch.tensor([0.7, 0.1, 0.1, 0.1])
        weighted_total, _ = loss_fn(pred, true, freq_weights=weights)

        assert weighted_total.item() > flat_total.item()

    def test_batched_multistep_shape(self):
        """(batch, lead_time, num_freqs, 7) — the shape a decoder output
        would actually have — must work, not just the bare (num_freqs, 7)
        used in the other tests here."""
        loss_fn = DirectionalLoss()
        batch, lead_time, num_freqs = 3, 5, 6
        true = torch.stack([
            _make_sample(1.5, 45.0, 200.0, 0.6, num_freqs=num_freqs)
            for _ in range(batch * lead_time)
        ]).reshape(batch, lead_time, num_freqs, 7)
        pred = true + 0.1

        total, components = loss_fn(pred, true)
        assert total.dim() == 0
        for v in components.values():
            assert v.dim() == 0


def _spike(freqs_len, center_idx, height=10.0, floor=1e-3):
    """A log-space single-bin spike (all other bins at `floor`) — a
    convenient narrow synthetic 'peak' for testing position-shift behavior."""
    x = np.full(freqs_len, floor)
    x[center_idx] = height
    return torch.log(torch.tensor(x, dtype=torch.float32))


class TestSpectralWassersteinLoss:
    """SpectralWassersteinLoss (nn/training_loop.py's Wasserstein-distance
    auxiliary term / nn/evaluate.py's always-on 'Shape_Wasserstein' metric)
    — the properties below are what distinguish it from a pointwise loss
    like RMSELoss: forgiving of small position shifts, mass-scale invariant
    (it compares shape, not magnitude)."""

    def test_zero_for_identical_spectra(self):
        loss_fn = SpectralWassersteinLoss()
        true = _spike(len(SPECTRAL_FREQS), 20)
        freqs = torch.tensor(SPECTRAL_FREQS)
        loss = loss_fn(true, true, freqs)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_mass_scale_invariant(self):
        """Scaling a prediction by a constant factor (adding a constant in
        log-space) must not change the loss — it compares normalized shape,
        not magnitude, by design (mass error is tracked separately by
        Shape_Mass_Error)."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        true = _spike(len(SPECTRAL_FREQS), 20)
        pred = _spike(len(SPECTRAL_FREQS), 30)  # shifted, so loss isn't trivially 0

        loss_unscaled = loss_fn(pred, true, freqs).item()
        loss_scaled = loss_fn(pred + np.log(5.0), true, freqs).item()  # 5x in linear space
        assert loss_scaled == pytest.approx(loss_unscaled, rel=1e-4)

    def test_forgiving_of_small_shifts_unlike_pointwise_error(self):
        """A narrow peak shifted by a small vs. large number of bins: W1
        must scale smoothly/monotonically with shift distance, while a
        plain pointwise error on the same pairs is roughly CONSTANT (a
        narrow peak has near-total non-overlap even for a 1-bin shift, so
        pointwise error can't distinguish a near miss from a far one) —
        this is the exact property motivating W1 over RMSELoss/the
        (reverted) SpectralSlopeLoss for the multimodal-blur problem."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = _spike(n, 20)

        def pointwise_mse(pred_log):
            pred_n = torch.exp(pred_log) / torch.exp(pred_log).sum()
            true_n = torch.exp(true) / torch.exp(true).sum()
            return ((pred_n - true_n) ** 2).mean().item()

        w1_near = loss_fn(_spike(n, 21), true, freqs).item()   # shift by 1 bin
        w1_mid = loss_fn(_spike(n, 25), true, freqs).item()    # shift by 5 bins
        w1_far = loss_fn(_spike(n, 45), true, freqs).item()    # shift by 25 bins

        assert w1_near < w1_mid < w1_far  # W1 scales with distance

        mse_near = pointwise_mse(_spike(n, 21))
        mse_far = pointwise_mse(_spike(n, 45))
        assert mse_near == pytest.approx(mse_far, rel=0.05)  # pointwise error is blind to distance

    def test_respects_nonuniform_grid(self):
        """A uniform (constant-height) true and pred spectrum on a
        deliberately uneven grid must still give zero loss (sanity: the
        cumulative integration doesn't introduce spurious error from grid
        non-uniformity alone), and a perturbed pred must give a loss that
        changes with WHERE on the grid the perturbation sits, not just its
        bin-index position — a pure index-based (non-Δf-aware) computation
        would be insensitive to this."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor([0.02, 0.03, 0.07, 0.08, 0.20, 0.485])
        flat_true = torch.log(torch.full((6,), 1.0))
        assert loss_fn(flat_true, flat_true, freqs).item() == pytest.approx(0.0, abs=1e-5)

        pred_a = torch.log(torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, 1.0]))  # bump in a narrow bin
        pred_b = torch.log(torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 1.0]))  # bump in a wide bin
        loss_a = loss_fn(pred_a, flat_true, freqs).item()
        loss_b = loss_fn(pred_b, flat_true, freqs).item()
        assert loss_a != pytest.approx(loss_b, rel=1e-3)

    def test_batched_multistep_shape(self):
        """(batch, lead_time, num_freqs) — the shape train_one_epoch's
        y_pred/y_batch actually have — must work, not just a bare 1-D
        spectrum."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        batch, lead_time = 3, 4
        true_1d = _spike(len(SPECTRAL_FREQS), 20)
        true = true_1d.expand(batch, lead_time, -1)
        pred = true.clone()

        loss = loss_fn(pred, true, freqs)
        assert loss.dim() == 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_reduction_none_matches_reduction_mean(self):
        """reduction='none' (used by nn/evaluate.py's 'density' block, which
        must mask out near-zero-mass samples before averaging) must return
        the same per-sample values that reduction='mean' (the training-loss
        default) averages over — not a different computation path."""
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        n = len(SPECTRAL_FREQS)
        true = torch.stack([_spike(n, 15), _spike(n, 20), _spike(n, 30)])
        pred = torch.stack([_spike(n, 16), _spike(n, 20), _spike(n, 35)])

        per_sample = loss_fn(pred, true, freqs, reduction='none')
        scalar = loss_fn(pred, true, freqs, reduction='mean')

        assert per_sample.shape == (3,)
        assert per_sample.mean().item() == pytest.approx(scalar.item(), rel=1e-5)
        # Identical pair (index 1) must be exactly zero; the others must not.
        assert per_sample[1].item() == pytest.approx(0.0, abs=1e-5)
        assert per_sample[0].item() > 0.0
        assert per_sample[2].item() > 0.0

    def test_reduction_invalid_raises(self):
        loss_fn = SpectralWassersteinLoss()
        freqs = torch.tensor(SPECTRAL_FREQS)
        true = _spike(len(SPECTRAL_FREQS), 20)
        with pytest.raises(ValueError, match="reduction"):
            loss_fn(true, true, freqs, reduction='sum')
