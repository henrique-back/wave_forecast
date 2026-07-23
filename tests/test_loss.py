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

from utils.loss import DirectionalLoss, _FULL_CHANNELS


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
